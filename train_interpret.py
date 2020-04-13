import train_linear_vrnn
import interpretion
from data_apis.SWDADialogCorpus import SWDADialogCorpus

ckpt_dir, ckpt_name = train_linear_vrnn.main(["--train"])
print("ckpt_dir: %s" % ckpt_dir)
print("ckpt_name: %s" % ckpt_name)

train_linear_vrnn.main(
    ["--decode", "--ckpt_dir", ckpt_dir, "--ckpt_name", ckpt_name])

interpretion.main(["--ckpt_dir", ckpt_dir, "--ckpt_name", ckpt_name])
