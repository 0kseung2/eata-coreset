"""
Copyright to EATA ICML 2022 Authors, 2022.03.20
Based on Tent ICLR 2021 Spotlight. 
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import math
import torch.nn.functional as F

from ptflops import get_model_complexity_info


class EATA(nn.Module):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, fishers=None, fisher_alpha=2000.0, steps=1, episodic=False, e_margin=math.log(1000)/2-1, d_margin=0.05, coreset_size=64):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = e_margin # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = d_margin # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

        self.fishers = fishers # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        self.fisher_alpha = fisher_alpha # trade-off \beta for two losses (Eqn. 8) 

        self.total_coreset_samples = 0
        self.total_coreset_sample_per_corruption = 0
        self.coreset_size = coreset_size

        # 받은 MACs를 처리하여 숫자와 단위를 분리해 변환
        # FLOPs 계산 및 출력
        input_shape = (3, 224, 224)  # ResNet50의 기본 입력 크기
        with torch.cuda.device(0):  # GPU 환경(가능할 경우)
            macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=False)
            value, unit = macs.strip().split() # '4.13', 'GMac'
            value = float(value)
            flops = value * 2  # MACs를 FLOPs로 변환
            print(f"Model: EATA \nFLOPs: {flops} GFLOPs, MACs: {macs}, Parameters: {params}")

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        if self.steps > 0:
            for _ in range(self.steps):
                outputs, num_counts_2, num_counts_1, updated_probs, coreset_samples_num = forward_and_adapt_eata(x, self.model, self.optimizer, self.fishers, self.e_margin, self.current_model_probs, fisher_alpha=self.fisher_alpha, num_samples_update=self.num_samples_update_2, d_margin=self.d_margin, coreset_size=self.coreset_size)
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)
                self.total_coreset_samples += coreset_samples_num # 코어셋 샘플 개수 누적
                self.total_coreset_sample_per_corruption += coreset_samples_num # 코어셋 샘플 개수 누적
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_eata(x, model, optimizer, fishers, e_margin, current_model_probs, fisher_alpha=50.0, d_margin=0.05, scale_factor=2, num_samples_update=0, coreset_size=64):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    Return: 
    1. model outputs; 
    2. the number of reliable and non-redundant samples; 
    3. the number of reliable samples;
    4. the moving average  probability vector over all previous samples
    """
    # forward
    outputs, features = model(x) # 순전파를 수행해서, 입력된 배치(x)에 대해서 각 샘플의 outputs를 만들어 낸다.
    # adapt
    entropys = softmax_entropy(outputs)
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    ids1 = filter_ids_1 # ex) 0, 3, 5, 6
    ids2 = torch.where(ids1[0]>-0.1) # ex) 0, 1, 2, 3이 됨 그래서 나중에 x[ids1][ids2]될 때, x[ids1]이랑 같게 됨.
    entropys = entropys[filter_ids_1]

    # filter redundant and coreset samples (신뢰도 필터링 -> 비중복 필터링 -> coreset 추출을 하는 것)
    if current_model_probs is not None: 
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin) # 비중복 샘플에서 필터링 된 인덱스
        entropys = entropys[filter_ids_2]
        ids2 = filter_ids_2
        k = min(coreset_size, filter_ids_2[0].size(0)) # coreset_size = 64, 32, 16, 8로 할것
        coreset_ids = dot_product_herding_coreset(features[filter_ids_1][filter_ids_2], k) # 두 번 필터링 된 샘플들에 대해서 코어셋으로 한 번 더 추출
        entropys = entropys[coreset_ids]
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2][coreset_ids].softmax(1))
    else:
        k = min(coreset_size, filter_ids_1[0].size(0)) # coreset_size = 64, 32, 16, 8로 할것
        coreset_ids = dot_product_herding_coreset(features[filter_ids_1], k) # 한 번 필터링 된 샘플들에 대해서 코어셋으로 한 번 더 추출
        entropys = entropys[coreset_ids]
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][coreset_ids].softmax(1))

    coreset_samples_num = len(coreset_ids) # 현재 배치에서 추출된 코어셋 샘플 개수

    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
    # implementation version 1, compute loss, all samples backward (some unselected are masked)
    entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
    loss = entropys.mean(0)
    """
    # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
    # if x[ids1][ids2].size(0) != 0:
    #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
    """
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
        loss += ewc_loss
    if x[ids1][ids2][coreset_ids].size(0) != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs, coreset_samples_num

def dot_product_herding_coreset(features, k):
    """
    내적 기반 herding coreset 알고리즘즘
    """
    # Herding-based coreset selection: 순차적으로 평균과의 차이를 줄이도록 샘플 선택
    
    device = features.device
    sample_num, features_num = features.shape
    mu = features.mean(dim=0)  # 전체 평균
    mu_t = torch.zeros(features_num, device=device)

    selected = []
    selected_mask = torch.zeros(sample_num, dtype=torch.bool, device=device) # 배열의 성분값들은 0(False)로 초기화한다. [False, False, False, False, ...]

    for t in range(1, k+1):
        remaining = ~selected_mask
        scores = torch.matmul(features[remaining], t * mu - (t - 1) * mu_t) # 내적 기반
        max_idx = torch.argmax(scores)
        true_idx = torch.nonzero(remaining, as_tuple=False)[max_idx].item()

        selected.append(true_idx)
        selected_mask[true_idx] = True

        # 평균 업데이트
        x_t = features[true_idx] # t번째 시점에서 선택된 feature vector
        mu_t = mu_t + (1/t) * (x_t - mu_t)

    return torch.tensor(selected, device=device)


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with eata."""
    # train mode, because eata optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what eata updates
    model.requires_grad_(False)
    # configure norm for eata updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with eata."""
    is_training = model.training
    assert is_training, "eata needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "eata needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "eata should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "eata needs normalization for its optimization"




