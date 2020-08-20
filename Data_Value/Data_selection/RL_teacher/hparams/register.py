_HPARAMS = dict()


def register(name):

  def add_to_dict(fn):
    global _HPARAMS
    _HPARAMS[name] = fn
    return fn

  return add_to_dict


def get_hparams(name):
  """Fetches a merged group of hyperparameter sets (chronological priority)."""
  return _HPARAMS[name]



