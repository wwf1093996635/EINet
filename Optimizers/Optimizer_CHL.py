


class Optimizer_CHL:
    def train(self, x, y): # x: input, y: label.
        out_free, act_free = self.model.forward_free(x)
        out_clamp, act_clamp = self.model.forward_clamp(x, y)
        self.model.update(act_free, act_clamp)
    
    