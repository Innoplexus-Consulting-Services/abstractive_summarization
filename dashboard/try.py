import subprocess

command = 'bazel-bin/textsum/seq2seq_attention   --mode=decode      \
    --article_key=article   --abstract_key=abstract                 \
    --data_path=data/test_1111_.bin   --vocab_path=data/vocab_1111_ \
    --log_root=log_root   --decode_dir=log_root/decode1111_         \
    --truncate_input=True --beam_size=20'

subprocess.call([command])
