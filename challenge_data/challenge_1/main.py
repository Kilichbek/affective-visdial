import random
import json
import nltk
from evaluate import load
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """

    # open files
    with open(test_annotation_file, "r") as f:
        test_annotation = json.load(f)

    print(test_annotation)
    with open(user_submission_file, "r") as f:
        user_submission = json.load(f)
        print(user_submission)
    dialogs = {d['dialog_id']: { "emotion": d["predicted_emotion"], "explanation": d["generated_explanation"]} for d in test_annotation}
    print(dialogs)
    ref_emotions = []
    ref_explanations = []
    pred_emotions = []
    pred_explanations = []
    pred_dialog_ids = []
    
    for d in user_submission:
        print(d)
        dialog_id = d["dialog_id"]
        pred_dialog_ids.append(dialog_id)
        ref_emotions.append(dialogs[dialog_id]["emotion"])
        ref_explanations.append(dialogs[dialog_id]["explanation"])
        pred_emotions.append(d["predicted_emotion"])
        pred_explanations.append(d["generated_explanation"])

    print("pred_explanations", pred_explanations)
    print("ref_explanations", ref_explanations)

    output = {}
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")

        precision, recall, f1, _ = precision_recall_fscore_support(ref_emotions, pred_emotions, average='weighted')
        f1 = f1 * 100

        bleu = load("bleu")
        bleu_results = bleu.compute(predictions=pred_explanations, references=ref_explanations, tokenizer=word_tokenize)

        bertscore = load("bertscore")
        bertscore_results = bertscore.compute(predictions=pred_explanations, references=ref_explanations, lang="en")
        bert_score = sum(bertscore_results['recall']) / len(ref_explanations)

        output["result"] = [
            {
                "train_split": {
                    "Metric1": f1,
                    "Metric2": bleu_results['bleu'],
                    "Metric3": bert_score,
                    "Total": (f1 + bleu_results['bleu'] + bert_score) / 3,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        precision, recall, f1, _ = precision_recall_fscore_support(ref_emotions, pred_emotions, average='weighted')
        f1 = f1 * 100

        bleu = load("bleu")
        bleu_results = bleu.compute(predictions=pred_explanations, references=ref_explanations, tokenizer=word_tokenize)

        bertscore = load("bertscore")
        bertscore_results = bertscore.compute(predictions=pred_explanations, references=ref_explanations, lang="en")
        bert_score = sum(bertscore_results['recall']) / len(ref_explanations)

        output["result"] = [
            {
                "test_split": {
                    "Metric1": f1,
                    "Metric2": bleu_results['bleu'],
                    "Metric3": bert_score,
                    "Total": (f1 + bleu_results['bleu'] + bert_score) / 3,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    return output
