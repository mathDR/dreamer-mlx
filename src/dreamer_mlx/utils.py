import mlx.core as mx


def symlog(x):
    return mx.sign(x) * mx.log(mx.abs(x) + 1.0)

def symexp(x):
    return mx.sign(x) * (mx.exp(mx.abs(x)) - 1.0)

class TwoHotDist:
    def __init__(self, logits, min_val=-20.0, max_val=20.0, num_bins=255):
        self.logits = logits
        self.bins = mx.linspace(min_val, max_val, num_bins)

    def mean(self):
        probs = mx.softmax(self.logits, axis=-1)
        return mx.sum(probs * self.bins, axis=-1)
