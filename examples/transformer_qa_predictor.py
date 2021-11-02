import timeit

import torch
import logging

from torch.utils.data import SequentialSampler, DataLoader
from transformers.data.metrics.squad_metrics import compute_predictions_log_probs, compute_predictions_logits, \
    squad_evaluate
from transformers.data.processors.squad import SquadResult, squad_convert_examples_to_features, SquadV1Processor, \
    SquadV2Processor

from yonlu.examples.transformer_qa_trainer import MODEL_CLASSES
from yonlu.question_answering.qa_trainer import QuestionAnsweringTrainer, to_list

import os

logger = logging.getLogger(__name__)

def predict(model, tokenizer, n_best_size, max_answer_length, version_2_with_negative,
            data_dir, predict_file,max_seq_length, doc_stride, max_query_length):
    #tokenizer, version_2_with_negative, data_dir, predict_file,max_seq_length, doc_stride, max_query_length

    dataset, examples, features = load_examples(tokenizer, version_2_with_negative,
                                                data_dir, predict_file,max_seq_length, doc_stride, max_query_length)

    pred_batch_size = 8

    # Note that DistributedSampler samples randomly
    pred_sampler = SequentialSampler(dataset)
    pred_dataloader = DataLoader(dataset, sampler=pred_sampler, batch_size=pred_batch_size)

    # Prediction!
    logger.info("***** Running prediction {} *****".format('pred'))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", pred_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in pred_dataloader:
        model.eval()
        batch = tuple(t.to('cpu') for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            start_logits = to_list(outputs[0][i])
            end_logits = to_list(outputs[1][i])
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join('./', "predictions_{}.json".format('pred'))
    output_nbest_file = os.path.join('./', "nbest_predictions_{}.json".format('pred'))

    output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        False,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,
        False,
        False,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


def load_examples(tokenizer, version_2_with_negative, data_dir, predict_file,
                  max_seq_length, doc_stride, max_query_length):

    # Load data features from cache or dataset file
    input_dir = "."

    logger.info("Creating features from dataset file at %s", input_dir)

    processor = SquadV2Processor() if version_2_with_negative else SquadV1Processor()

    examples = processor.get_dev_examples(data_dir, filename=predict_file)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )

    return dataset, examples, features



if __name__ == "__main__":
    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}

    logger.info("Loading checkpoint %s for evaluation", 'monologg/kobert')

    config_class, model_class, tokenizer_class = MODEL_CLASSES['kobert']
    config = config_class.from_pretrained(
        './models/checkpoint-4000/',
        cache_dir=None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        'monologg/kobert',
        do_lower_case=False,
        cache_dir=None,
    )
    model = model_class.from_pretrained(
        './models/checkpoint-4000/',
        from_tf=bool(".ckpt" in 'monologg/kobert'),
        config=config,
        cache_dir=None,
    )

    qa_trainer = QuestionAnsweringTrainer()

    n_best_size = 20
    max_answer_length = 30
    version_2_with_negative = False
    data_dir = 'D:\\python_workspace\\yonlu\\yonlu\\data'
    predict_file = 'dev-evaluate-in1.json'
    max_seq_length = 512
    doc_stride = 128
    max_query_length = 64

    model.to('cpu')

    # Evaluate
    result = predict(model, tokenizer, n_best_size, max_answer_length, version_2_with_negative,
                     data_dir, predict_file,max_seq_length, doc_stride, max_query_length)

    result = dict((k + ("_1"), v) for k, v in result.items())
    results.update(result)

    logger.info("Results: {}".format(results))
