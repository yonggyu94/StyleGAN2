import os


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def make_folder(exp_dir, log_dir, model_dir, sample_dir):
    log_root = os.path.join(exp_dir, log_dir)
    model_root = os.path.join(exp_dir, model_dir)
    sample_root = os.path.join(exp_dir, sample_dir)

    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    if not os.path.isdir(log_root):
        os.makedirs(log_root, exist_ok=True)
    if not os.path.isdir(model_root):
        os.makedirs(model_root, exist_ok=True)
    if not os.path.isdir(sample_root):
        os.makedirs(sample_root, exist_ok=True)

    return log_root, model_root, sample_root
