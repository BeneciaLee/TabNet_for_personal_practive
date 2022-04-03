# This project was created using the following address.
# https://www.kaggle.com/code/samratthapa/tabnet-implementation?scriptVersionId=46472520

# Arık, "Tabnet: Attentive interpretable tabular learning."
# https://arxiv.org/abs/1908.07442v4
import torch
import torch.nn as nn

# Tabnet은 스텝이 진행됨에 따라서 집중하는 피처가 달라진다.
# 이러한 특성을 이용해서 한 번의 스텝동안에 중요한 피처에 집중하게 된다.
# 예를 들어서 10개의 피처가 있다고 하면 기존의 MLP는 모든 피쳐를 동시에 본다.
# 하지만 Tabnet은 Mask 기법을 통해서 모든 피쳐를 한번에 보지 않고 스텝당 나눠서 따로 집중해서 본다.
# 이러한 Mask를 생성하기 위해서 사용되는 것이 Sparsemax이다.
# Mask 생성을 위해서 기존의 nn.Softmax를 사용하지 않는다.
# 왜냐하면 하나의 피쳐에 집중하기 위해서는 기존의 Softmax보다 Sparsemax가 적합하다.
# Sparsemax는 확률 분포가 중요한 특징들을 1에 더 가깝게 출력하게 하고 중요하지 않는 특징들은 0에 더 가깝게 출력하는 특징이 있다.
class Sparsemax(nn.Module):
    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        number_of_logits = input.size(dim)

        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, device=device, step=1, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]
        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)
        self.output = torch.max(torch.zeros_like(input), input - taus)
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)
        return output

    def backward(self, grad_output):
        dim = 1
        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))
        return self.grad_input

# Ghost Batch Normalization(GBN)
# vbs값에 따라서 그룹을 만들고 그룹내에서 BatchNorm1d를 진행한다.
# Batchsize가 애초에 128보다 작은 값이라면 GBN을 하는 의미가 없어 보인다.
# GBN은 큰 배치 사이즈에서 사용하는 것이 적합하다고 판단된다.
# 실제 real-data 같은 경우는 정말 많은 데이터들이 들어오기 때문에 GBN을 통해서 배치 정규화를 하는 것이 좋아 보인다.
class GBN(nn.Module):
    def __init__(self,inp,vbs=128,momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp,momentum=momentum)
        self.vbs = vbs
    def forward(self,x):
        # split과 다른 점은 split은 인자로 들어오는 값을 통해서 그룹을 묶고
        # chunk는 인자로 들어온 숫자 만큼 그룹을 만든다.
        chunk = torch.chunk(x,x.size(0)//self.vbs,0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res,0)

# AttentiveTransformer를 통해서 Mask를 생성한다.
#
class AttentiveTransformer(nn.Module):
    def __init__(self, d_a, inp_dim, relax, vbs = 128):
        super().__init__()
        # FeatureTransformer로 부터 넘어노는 a를 입력으로 받는다.
        # a는 d_a 크기의 벡터 사이즈를 갖고 있기 때문에 우리가 갖고 있는 데이터에 맞게
        # input_size로 project 해준다.
        # 이렇게 되면 기존의 데이터와 동일한 크기의 벡터가 생성한다.
        # |original_data| = |self.fc(x)| = (bs, #features)
        self.fc = nn.Linear(d_a, inp_dim)
        self.bn = GBN(inp_dim, vbs = vbs)
        self.smax = Sparsemax()
        self.r = relax

    def forward(self, a, priors):
        # 마스크를 만들기 위해서 프로젝션을 수행한다.
        # 이를 통해서 기존의 데이터와 동일한 크기를 갖는 텐서 모양이 만들어진다.
        # 기존과 같은 텐서 모양이 만들어 졌다면 이를 다시 배치 정규화를 수행해준다.
        # 여기서 입력으로 받은 a는 FeatureTransformer의 출력 값이다. -> FeatureTransformer의 출력 값으로는 a와 d가 존재한다.
        # a는 다음 스텝들의 마스크를 만들기 위해서 계속 다음 스텝으로 넘어간다.
        # 넘어 갈때 다음 스텝의 마스크를 만들기 쉽게 하는 벡터 정보가 들어있을 것이다.
        a = self.bn(self.fc(a))
        # 위의 a와 priors를 곱해서 smax의 인자값으로 넣는다.
        # 이를 통해서 마스크에 해당하는 확률 분포를 생성한다.
        # priors의 역할은 이전에 수행했던 마스크들에 대한 정보가 들어있다.
        # 즉, 이전에 집중해서 보았던 피쳐들은 무엇이 있었고 앞으로 집중해야 하는 피쳐가 무엇인지를 포함하고 있다.
        # 이번에 사용했던 피쳐는 앞으로의 스텝에서 사용하지 않도록 해주고 사용하지 않았던 피쳐에 대해서는 집중할 수 있도록 도와준다.
        # 아래와 같이 mask를 만드는 트릭을 기억해두는 것이 좋아보인다. 다른 곳에도 응용하기 좋을 것 같다.
        mask = self.smax(a * priors)
        priors = priors * (self.r - mask)
        return mask

# 논문을 읽으면서 GLU가 무엇인지를 몰랐다.
# 하지만 코드를 보니 GLU가 무엇을 의도하는지 조금은 알 것 같다.
class GLU(nn.Module):
    def __init__(self, inp_dim, out_dim, fc = None, vbs = 128):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim, out_dim * 2)
        # 개인적으로는 작은 데이터 셋에서는 GBN을 사용할 필요가 없을 것 같다.
        # 애초에 데이터가 작기 때문에 batchsize를 크게 가져갈 수도 없고
        # 적은 batch_size를 사용하게 된다면 nn.BatchNorm1d를 바로 사용해도 되기 때문이다.
        # 하지만 논문에서는 batch_size를 크게 가져갈 수 있는 데이터를 사용했기 때문에
        # 다음과 같이 GBN을 사용한 것 같다.
        # 논문에서는 Rossmann 에서는 B = 4096, KDD 에서는 B = 8192, Mushroom edibility에서는 B = 2048, Adult Census Income
        # 에서는 B = 4096을 사용했다.
        # 그리고 다음과 같이 논문저가가 batch_size 설정을 추천하고 있다.
        # A large batch size is beneficial – if the memory constraints permit,
        # as large as 1-10 % of the total training dataset size can help performance.
        # The virtual batch size is typically much smaller
        self.bn = GBN(out_dim * 2, vbs= vbs)
        self.od = out_dim

    def forward(self, x):
        x = self.bn(self.fc(x))
        # 이를 통해서 지속적으로 집중해야 하는 피쳐가 무엇인지를 판별하고 전송한다.
        # 밑의 return 값의 의도를 파악하는 것이 GLU의 핵심인 것 같다.
        # 나는 단순히 논문을 읽을 때는 GLU가 대체 뭔데??? 하고 물음표가 생겼지만, 코드를 보니 저자가 무엇을
        # 의도했는지를 어렴풋이 이해할 수 있었다.
        # 아래와 같은 방식도 기억해두는 것이 좋아 보인다. Tabnet을 통해서 새롭게 배우는 트릭들이 많아서 재밌다.
        return x[:, :self.od] * torch.sigmoid(x[:, self.od:])

class FeatureTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs= 128):
        super(FeatureTransformer, self).__init__()
        # 사실 Tabnet을 공부하기 이전에는 ModuleList를 사용해본 적이 없다.
        # 주로, nn.Sequential()를 통해서 모델을 구성하고 설계했다.
        # 하지만 다음과 같이 모델을 구성할 때는 ModuleList가 더욱 유리해 보인다. 편하기도 하다.
        self.shared_layers = nn.ModuleList()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if shared:
            # Tabnet에서 FeatureTransformer의 구성을 보면 최초의 2개는 shared한 레이어이고
            # 뒤의 2개가 step 마다 달라지는 레이어라는 것을 알 수 있다.
            # shared한 레이어라는 것은 가중치를 공유한 레이어를 의미한다.
            # 그리고 최초로는 Mask와 곱해진 입력 값이 들어오기 때문에 (bs, #features)가 된다.
            # 이러한 입력 값은 (bs, d_a + d_d)의 크기로 변해서 출력된다.
            self.shared_layers.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            for fc in shared[1:]:
                self.shared_layers.append(GLU(out_dim, out_dim, fc, vbs = vbs))

        self.depend_layers = nn.ModuleList()
        for x in range(n_ind):
            # depend_layers는 step마다 다른 값들을 갖고 있다.
            # 이를 통해서 최종 출력 값과 마스크를 만들기 위해서 필요한 벡터 값들이 생성된다.
            self.depend_layers.append(GLU(out_dim, out_dim, vbs = vbs))
        self.scale = torch.sqrt(torch.tensor([.5], device = self.device))


    def forward(self, x):
        # |x| = (bs, #features)
        if self.shared_layers:
            # Tabnet 논문을 보면 최초의 레이어에서는 잔차 연결을 수행하지 않는다.
            # 여기에서는 다음 레이어로 바로 벡터 값들 넘기고
            # 그 다음부터는 잔차 연결을 모든 레이어에서 수행한다.
            x = self.shared_layers[0](x)
            for glu in self.shared_layers[1:]:
                x = torch.add(x, glu(x)) # x = x + glu(x) 와 동일하다.
                x = x * self.scale

        for glu in self.depend_layers:
            # depend_layers에서는 모든 레이어에서 잔차 연결을 수행한다.
            # 논문의 그림을 보면 이해가 갈 것이다.
            x = torch.add(x, glu(x))
            x = x * self.scale

        # |x| = (bs, d_d + d_a)
        return x

class DecisionStep(nn.Module):
    def __init__(self, inp_dim, d_d, d_a, shared, n_dep, relax, vbs = 128):
        super(DecisionStep, self).__init__()
        self.fe_trans = FeatureTransformer(inp_dim, d_d + d_a, shared, n_dep, vbs)
        self.at_trans = AttentiveTransformer(d_a, inp_dim, relax, vbs)

    def forward(self, x, a, priors):
        # 각 결정 단계에서는 3개의 인자들이 필요하다.
        # 모든 step에서는 동일한 x를 인자로 받는다. |x| = (batch_size, #features)
        # a는 FeatureTransformer로 부터 생성된 a를 말한다.
        # priors는 AttentiveTransforemr에서 마스크를 생성하기 필요한 인자 값이다.
        mask = self.at_trans(a, priors)
        x = self.fe_trans(x * mask)
        return x, mask

class TabNet(nn.Module):
    def __init__(self, inp_dim, final_out_dim, d_d, d_a, n_shared, n_dep,
                 n_steps, relax = 1.5, vbs = 128):
        super().__init__()
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (d_d + d_a)))
            for x in range(n_shared - 1):
                self.shared.append((nn.Linear(d_d + d_a, 2 * (d_d + d_a))))

        # kaggle 코드에서는 처음 부터 step이라고 잡지만 나는 attentive가 같이 있는 DecisionStep 부터 step이라고 잡았다.
        # 그냥 논문의 그림과 똑같이 맞춰주기 위해서이다.
        self.zero_step = FeatureTransformer(inp_dim, d_d + d_a, self.shared, n_dep)

        self.steps = nn.ModuleList()
        for x in range(n_steps):
            self.steps.append(DecisionStep(inp_dim, d_d, d_a, self.shared, n_dep, relax, vbs))
        self.fc = nn.Linear(d_d, final_out_dim)
        self.bn = nn.BatchNorm1d(inp_dim)
        self.d_d = d_d

    def forward(self, x):
        # 최초의 입력 값은 배치 정규화를 시켜서 스케일 조정을 수행한다.
        x = self.bn(x)
        # zero_step에 넣어서 다음 step 부터 사용해야 하는 a를 생성한다.
        # [ d ; a ] 로 이루어져 있다.
        x_a = self.zero_step(x)[:, self.d_d:]
        mask = []
        out = torch.zeros(x.size(0), self.d_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te, m = step(x, x_a, priors)
            mask.append(m)
            out += nn.functional.relu(x_te[:, :self.d_d])
            x_a = x_te[:, self.d_d:]

        # Tabnet을 공부하면서 Transformer와 같이 배우는 부분들이 많아서 재밌는 것 같다.
        # 앞으로 이것을 어떻게 더 변형 시켜서 사용할 수 있는지를 생각해 볼 필요가 있다고 생각한다.
        return self.fc(out), mask

