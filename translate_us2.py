import argparse
import logging
import sys
import random
import torch
from src.translator import Translator
from utils.vocabulary import collect_vocabularies
from src.serialize import load_model
from src.word_by_word import WordByWordModel


def translate_opts(parser):
    group = parser.add_argument_group('Vocabulary')
    group.add_argument('-src_vocabulary', default="src.pickle",
                       help="Path to source vocab")
    group.add_argument('-tgt_vocabulary', default="tgt.pickle",
                       help="Path to target vocab")
    group.add_argument('-all_vocabulary', default="all.pickle",
                       help="Path to all vocab")

    # Embedding Options
    group = parser.add_argument_group('Embeddings')
    group.add_argument('-src_embeddings', type=str, default='data/vec/vectors-en.txt',
                       help='Pretrained word embeddings for source language.')
    group.add_argument('-tgt_embeddings', type=str, default='data/vec/vectors-bn.txt',
                       help='Pretrained word embeddings for target language.')
                       
                       
    # Word by word
    group.add_argument('-src_to_tgt_dict', type=str, default='data/en-bn.txt',
                       help='Source[ENG] to Target[BN] Dictionary.')
    group.add_argument('-tgt_to_src_dict', type=str, default='data/bn-en.txt',
                       help='Target[BN] to Source[ENG] Dictionary')
    group.add_argument('-max_length', type=int, default=75,
                       help="Sentence max length")
    group.add_argument('-seed', type=int, default=42,
                       help="""Random seed used for the experiments reproducibility.""")
     
    group = parser.add_argument_group('Model')
    group.add_argument('-model', type=str, default='trained_model.pt',help='Path to model .pt file')
    group.add_argument('-s', default='data/eng_test2.txt', help='source_file')
    group.add_argument('-sl', type=str, default="e",help='source_language(either e or b)')
    group.add_argument('-t', default='predicted_output.txt',help='target_file name')
    
    

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts
translate_opts(parser)
opt = parser.parse_args()
random.seed(opt.seed)

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    use_cuda = torch.cuda.is_available()
    logging.info("Use CUDA: " + str(use_cuda))
  
    _, _, vocabulary = collect_vocabularies(
            src_vocabulary_path=opt.src_vocabulary,
            tgt_vocabulary_path=opt.tgt_vocabulary,
            all_vocabulary_path=opt.all_vocabulary, 
            reset=False)
    if opt.src_to_tgt_dict is not None and opt.tgt_to_src_dict is not None:
        translator = WordByWordModel(opt.src_to_tgt_dict, opt.tgt_to_src_dict, vocabulary, opt.max_length)
    else:
        model, _, _, _ = load_model(opt.model, use_cuda)
        translator = Translator(model, vocabulary, use_cuda)
    input_filename = opt.s
    output_filename = opt.t
    
    if opt.sl == 'e':
        lang = 'src'
    else:
        lang = 'tgt'
  
    tgt_lang = "src" if lang == "tgt" else "tgt"
    
    logging.info("Writing output...")
    with open(input_filename, "r", encoding="utf-8",errors="ignore") as r, open(output_filename, "w", encoding="utf-8",errors="ignore") as w:
        for line in r:
            translated = translator.translate_sentence(line, lang, tgt_lang)
            logging.debug(translated)
            w.write(translated+"\n")

if __name__ == "__main__":
    main()
