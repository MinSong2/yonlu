import datetime
import argparse
import yonlu.bert.bart_summarizer as summerizer
import logging

logger = logging.getLogger(__name__)

def do_summarize(contents):
    document = str(contents)
    logger.info("Document Created")

    doc_length = len(document.split())
    logger.info("Document Length: " + str(doc_length))

    min_length = int(doc_length/6)
    logger.info("min_length: " + str(min_length))
    max_length = min_length+200
    logger.info("max_length: " + str(max_length))

    transcript_summarized = summarizer.summarize_string(document, min_length=min_length, max_length=max_length)
    with open("summarized.txt", 'a+', encoding="utf-8") as file:
        file.write("\n" + str(datetime.datetime.now()) + ":\n")
        file.write(transcript_summarized + "\n")

parser = argparse.ArgumentParser(description='Summarization of text using CMD prompt')
parser.add_argument('--bart_checkpoint', default=None, type=str, metavar='PATH',
                    help='Path to optional checkpoint. Semsim is better model but will use more memory and is an additional 5GB download. (default: none, recommended: semsim)')
parser.add_argument('--bart_state_dict_key', default='model', type=str, metavar='PATH',
                    help='model state_dict key to load from pickle file specified with --bart_checkpoint (default: "model")')
parser.add_argument('--bart_fairseq', action='store_true',
                    help='Use fairseq model from torch hub instead of huggingface transformers library models. Can not use --bart_checkpoint if this option is supplied.')
parser.add_argument('--mode', default='ko', type=str, help='either Korean or English huggingface model')

args = parser.parse_args()

logger.info("Loading Model")

summarizer = summerizer.BartSumSummarizer(checkpoint=args.bart_checkpoint,
                        state_dict_key=args.bart_state_dict_key,
                        hg_transformers=(not args.bart_fairseq),
                        mode=args.mode)


f = open("../data/sample_data_summary_ko.txt", "r", encoding="UTF-8")
text = f.read()
f.close()

do_summarize(text)