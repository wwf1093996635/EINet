from Optimizers import * # make this folder a package
from Optimizers.Optimizer import Optimizer
from Optimizers.Optimizer_TP import Optimizer_TP # removing Optimizers. prefix results in error, since sys.path does not contain ./Models/
from Optimizers.Optimizer_BP import Optimizer_BP