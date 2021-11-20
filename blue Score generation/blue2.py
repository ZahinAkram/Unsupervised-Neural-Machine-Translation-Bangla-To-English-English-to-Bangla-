import os
import re
import subprocess
import tempfile
import logging

import numpy as np

from six.moves import urllib

logger = logging.getLogger('blue')


def get_moses_multi_bleu(hypotheses, references, lowercase=False):
    if isinstance(hypotheses, list):
        hypotheses = np.array(hypotheses)
    if isinstance(references, list):
        references = np.array(references)

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    # Get MOSES multi-bleu script
    try:
        multi_bleu_path, _ = urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
            "master/scripts/generic/multi-bleu.perl")
        os.chmod(multi_bleu_path, 0o755)
    except:
        logger.warning("Unable to fetch multi-bleu.perl script")
        return None

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile('pred.txt')
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush('pred_bleu.txt')
    reference_file = tempfile.NamedTemporaryFile('eng_test.txt')
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush('eng_test_bleu.txt')

    # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
            bleu_score = np.float32(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                logger.warning("multi-bleu.perl script returned non-zero exit code")
                logger.warning(error.output)
            bleu_score = None

    # Close temp files
    hypothesis_file.close()
    reference_file.close()

    return bleu_score