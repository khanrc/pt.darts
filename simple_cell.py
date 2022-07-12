import torch.nn as nn
from models.search_cells import SearchCell

class Simple_Cell(nn.Module):
    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        super().__init__()
        C = C * 2  # double such that after cells we have 512 dim size rather than 256 for ssd piping purposes
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        num_classes = 91

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        C_pp, C_p, C_cur = C_cur, C_cur, C

        # self.stem = torchvision.models.mobilenet_v2(pretrained=True).features
        # C_pp, C_p, C_cur = 1280, 1280, 128

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3, n_layers]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            print(f"cell{i} shape is {cell.preproc1.net[1]}")
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)  # use tensor form of images not transformed form
        # s0 = s1 = self.stem(images.tensors)
        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        return s1