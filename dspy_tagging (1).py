import ujson as json
import os
import pickle
import random
from enum import Enum

from tqdm import tqdm

import dspy
from dotenv import load_dotenv
from dspy import Example
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFewShot
from dsp.modules.hf_client import Anyscale
from dspy.teleprompt.ensemble import Ensemble
from preprocess_utils import concern_list
from config.definitions import ROOT_DIR
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import wandb
import re
import datetime
from wandb.sdk.data_types.trace_tree import Trace
from utils import get_null_files
# from dspy.primitives.assertions import Assert


# marvin.settings.openai.api_key = os.getenv('OPENAI_API_KEY')
load_dotenv()
concern_list = concern_list()
concern_set = set([_.lower().strip() for _ in concern_list])
concern_concat_string = ""
for concern in concern_list:
    concern_concat_string += concern + ",\n"
WANDB_BASE_URL = "https://api.wandb.ai"
os.environ["WANDB_BASE_URL"] = WANDB_BASE_URL
WB_PROJECT = "concern-detection"  # top-level directory for this work
WB_ENTITY = "darinkishore"  # user or team name
# modules

class MostLikelyConcernsSignature(dspy.Signature):
    """ Determines the FIVE most likely concerns in the list of possible concerns. FIVE."""
    title = dspy.InputField(desc="title of reddit post")
    post = dspy.InputField(desc="body of reddit post")
    possible_concerns = dspy.InputField(desc="list of possible concerns to pick from")
    most_likely_concerns = dspy.OutputField(
        desc="FIVE most probable concerns, separated by commas, ex: Concerns: depression, anxiety, ...")


# for each concern, determine if it is indeed present in the post
class ConcernPresentSignature(dspy.Signature):
    """Determines if a concern is an important factor in the post."""
    title = dspy.InputField(desc="title of reddit post")
    post = dspy.InputField(desc="body of reddit post")
    concern = dspy.InputField(desc="concern to check")
    reasoning = dspy.OutputField(desc="why the concern does or does not play an important role in the post")
    concern_present = dspy.OutputField(desc=f"TRUE if concern is present, FALSE if concern is not present")


# output the list of concerns that are present in the post

def clean_concerns_to_list(messy_concerns: str) -> list[str]:
    """
    Cleans messy concerns string to list of concerns
    :param messy_concerns:
    :return:
    """
    concerns = messy_concerns.split(',')
    pattern = re.compile(r'[Cc]on.+:( )?', re.IGNORECASE)
    concerns = [pattern.sub('', concern).strip() for concern in concerns]
    # filter by concerns in concern_list
    for concern in concerns:
        if concern.lower().strip() not in concern_set:
            concerns.remove(concern)
    return concerns


def true_or_false(string):
    false_pattern = re.compile(r'(?i)\bfalse\b')
    true_pattern = re.compile(r'(?i)\btrue\b')

    if false_pattern.search(string):
        return False
    elif true_pattern.search(string):
        return True
    else:
        return None


class DetectConcern(dspy.Module):
    def __init__(self):
        super().__init__()
        self.most_likely_concerns = dspy.Predict(MostLikelyConcernsSignature, max_tokens=100)
        self.concern_present = dspy.ChainOfThought(ConcernPresentSignature)

    def forward(self, title, post, possible_concerns, bypass_assert=False):
        root_span = Trace(
            name="ConcernDetectionAgent",
            kind="agent",
            metadata={"model": "turbo"}
        )

        # Get the most likely concerns
        most_likely_concerns = self.most_likely_concerns(
            title=title, post=post, possible_concerns=possible_concerns
        ).most_likely_concerns


        likely_span = Trace(
            name="likely_concerns",
            inputs={"title": title, "post": post},
            outputs={"most_likely_concerns": most_likely_concerns}
        )
        root_span.add_child(likely_span)

        # Process the concerns
        cleaned_concerns = clean_concerns_to_list(most_likely_concerns)
        detected_concerns = []
        # if not bypass_assert:
        #     dspy.Assert(
        #         len(cleaned_concerns) < 6,
        #         msg="You should have at most five concerns.",
        #     )
        # for first five concerns, check if they are present in the post
        for clean_concern in cleaned_concerns[:6]:
            concern_present = self.concern_present(
                title=title, post=post, concern=clean_concern
            )
            is_concern_present = concern_present.concern_present
            reasoning = concern_present.reasoning
            concern_span = Trace(
                name="concern_present",
                inputs={"concern": clean_concern, "title": title, "post": post},
                outputs={"concern_present": is_concern_present, "reasoning": reasoning}
            )
            root_span.add_child(concern_span)

            # if not bypass_assert:
            #     dspy.Assert(
            #         true_or_false(is_concern_present) is not None,
            #         msg="Make sure you output TRUE or FALSE after your reasoning.",
            #     )
            if true_or_false(is_concern_present):
                detected_concerns.append(clean_concern)

        detected_concerns = ', '.join(detected_concerns)

        root_span.add_inputs_and_outputs(
            inputs={"title": title, "post": post},
            outputs={"detected_concerns": detected_concerns})
        root_span.log(name="nice_concerns")

        return detected_concerns


# metrics

def concerns_to_vector(concerns, concern_list):
    return [1 if concern in concerns else 0 for concern in concern_list]


def eval_metrics(y_true, y_pred, concern_list):
    y_true_bin = [concerns_to_vector(entry, concern_list) for entry in y_true]
    y_pred_bin = [concerns_to_vector(entry, concern_list) for entry in y_pred]

    precision = precision_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average='samples')
    f1 = f1_score(y_true_bin, y_pred_bin, average='samples')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return precision, recall, f1, None, None


def evaluate_on_test_set(pipeline, test_examples, concern_list):
    y_true = [entry.detected_concerns.split(',') for entry in test_examples]
    y_pred = []
    eval_trace = Trace(
        name="Evaluation",
        kind="LLM",
        metadata={"model": "turbo", "timestamp": datetime.datetime.now()}
    )

    for entry in tqdm(test_examples):
        title = entry.title
        text = entry.post
        result = pipeline(title=title, post=text, possible_concerns=concern_concat_string)
        output = result
        test = Trace(
            name="EvaluationExample",
            kind='LLM',
            inputs={"title": title, "post": text, "possible_concerns": concern_concat_string},
            outputs={"detected_concerns": output})
        eval_trace.add_child(test)
        output = [concern.strip() for concern in output.split(',')]
        pattern = re.compile(r'[Cc]on.+:( )?')
        output = [pattern.sub('', concern).strip() for concern in output]

        y_pred.append(output)
    eval_trace.log(name="Evaluation_Completed")

    precision, recall, f1, macro_accuracy, exact_match_accuracy = eval_metrics(y_true, y_pred, concern_list)
    wandb.log({"precision": precision, "recall": recall, "f1": f1, "macro_accuracy": macro_accuracy,
               "exact_match_accuracy": exact_match_accuracy})


def partial_match_concern(example, prediction, trace=None):
    """
    Evaluates if the prediction is a close match to the example based on a given threshold of allowed differences.
    :param trace:
    :param example: Example object
    :param prediction: prediction
    :return: boolean
    """
    allowed_difference = 1
    actual_output = example.detected_concerns
    predicted_output = prediction  # .completions.detected_concerns[0]
    try:
        predicted_concerns = [concern.strip().lower() for concern in actual_output.split(',')]
        actual_concerns = [concern.strip().lower() for concern in predicted_output.split(',')]
        pattern = re.compile(r'concern*:', re.IGNORECASE)
        predicted_concerns = [pattern.sub('', concern).strip() for concern in predicted_concerns]
        predicted_concerns = set(predicted_concerns)
        actual_concerns = set(actual_concerns)
        # if predicted concerns is 3 bigger than actual concerns, return false
        if len(predicted_concerns) > len(actual_concerns) + 3:
            return False
        # TODO: find the crises that are most frequently simultaneously occurring in our dataset
        # for all of these crises, if there is one, add and remove both of them from the set
    except Exception as e:
        print(f"Failed to split actual or predicted output due to: {e}")
        return False

    # if every predicted concern is in the actual concerns, then it is a match
    for concern in actual_concerns:
        if concern not in predicted_concerns:
            allowed_difference -= 1

    if allowed_difference < 0:
        return False
    else:
        return True


# data/helper functions


def create_dataset(results):
    random.seed(42)
    # result is in format # dict, {"filename.json": (title, post, concerns)}
    examples = []
    for json_name, data in results.items():
        title = data[0]
        post = data[1]
        crises = data[2].strip()
        # create list of examples
        examples.append(
            Example(title=title, post=post,
                    possible_concerns=concern_concat_string, detected_concerns=crises)
            .with_inputs('title', 'post', 'possible_concerns'))
    random.shuffle(examples)

    # split into train, dev, copy examples for test
    test_examples = examples.copy()
    dev_examples = []
    train_examples = []
    flag = False

    # add "no concern examples"
    for example in examples:
        if example.detected_concerns == "NO CONCERN DETECTED":
            if not flag:
                dev_examples.append(example)
                examples.remove(example)
                flag = True
            else:
                train_examples.append(example)
                examples.remove(example)
                flag = False

    # add the rest in a 60/40 split, dev 60, train 40
    for example in examples:
        if len(dev_examples) < len(examples) * 0.6:
            dev_examples.append(example)
        else:
            train_examples.append(example)

    # randomly sample rest for train and dev

    random.shuffle(train_examples)
    random.shuffle(dev_examples)
    random.shuffle(test_examples)

    return train_examples, dev_examples, test_examples


def extract_data(json_folder):
    """
    Extracts conversation and identified crises
    """

    backup_results = get_null_files()

    results = {}
    for file in os.listdir(json_folder):
        if not file.endswith('.json'):
            continue
        file_path = os.path.join(json_folder + "/" + file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            title = data.get('title')
            post = data.get('post')
            if not post:
                for backup_file in backup_results:
                    if backup_file['sid'] == file[:-5]:
                        post = backup_file['text']
                        break
            concerns = data.get('concerns')
            results[file] = (title, post, concerns)
    # ensure all results have a title, post, and concerns
    for file in results:
        if not results[file][0] or not results[file][1] or not results[file][2]:
            print(f"Missing title, post, or concerns for {file}")
            del results[file]

    return results


def get_data():
    folder = os.path.join(ROOT_DIR, "dat", "college", "processed_gpt4")
    results = extract_data(folder)
    train_examples, dev_examples, test_examples = create_dataset(results)
    return train_examples, dev_examples, test_examples


# lm = Anyscale(model="meta-llama/Llama-2-13b-chat-hf", use_wandb=True, span_name="teleprompt",
# proj_name="concern-detection", max_tokens=200)

# pipeline
def compile_pipeline(model_name):
    """
    This function compiles the pipeline for concern detection.
    The function also saves the compiled pipeline to a pickle file and a json file.

    Args:
        model_name (str, optional): Name of the model. Defaults to "llama".

    Returns:
        tuple: The compiled pipeline and the test examples.
    """
    run = wandb.init(project=WB_PROJECT, entity=WB_ENTITY, save_code=True, tags=["zephyr-7b-beta"])


    RECOMPILE_INTO_LLAMA_FROM_SCRATCH = True

    metric_EM = partial_match_concern

    train_examples, dev_examples, test_examples = get_data()

    # lm = dspy.OpenAI(model=model_name, api_key=os.getenv('OPENAI_API_KEY'))
    # meta-llama/Llama-2-13b-hf meta-llama/Llama-2-13b-chat-hf
    # lm = dspy.HFClientTGI(model="meta-llama/Llama-2-chat-13b-hf", port=8080, url="http://localhost", max_tokens=400)
    lm = dspy.HFClientTGI(model="HuggingFaceH4/zephyr-7b-beta", port=[8080, 8081, 8082, 8083, 8084, 8085], url="http://localhost", max_tokens=400)
    # lm = Anyscale(model="meta-llama/Llama-2-70b-chat-hf", max_tokens=250)

    dspy.settings.configure(lm=lm)
    if RECOMPILE_INTO_LLAMA_FROM_SCRATCH:
        tp = BootstrapFewShot(metric=metric_EM)
        compiled_boostrap = tp.compile(DetectConcern(), trainset=train_examples[:100], valset=train_examples[101:])
        print("woof")
        # double = BootstrapFewShot(metric=metric_EM, max_bootstrapped_demos=2, max_rounds=1, max_labeled_demos=2)
        # compiled_detect_crises = double.compile(DetectConcern(), teacher=compiled_boostrap,
        # trainset=train_examples[:50], valset=train_examples[51:])
        try:
            compiled_boostrap.save(os.path.join(ROOT_DIR, "dat", "college", f"{model_name}_concerndetect.json"))
            # save a pickle file
            with open(os.path.join(ROOT_DIR, "dat", "college", f"{model_name}_concerndetect.pkl"), "wb") as f:
                pickle.dump(compiled_boostrap, f)
            artifact = wandb.Artifact(name=f"{model_name}-concern-detection", type="teleprompter")
            artifact.add_file(os.path.join(ROOT_DIR, "dat", "college", f"{model_name}_concerndetect.json"))
            artifact.add_file(os.path.join(ROOT_DIR, "dat", "college", f"{model_name}_concerndetect.pkl"))
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"Failed to save using compiled_detect_crises.save() due to: {e}")
        print("Evaluating on test set...")


    # if not RECOMPILE_INTO_LLAMA_FROM_SCRATCH:
        # try:
            # artifact = run.use_artifact('darinkishore/concern-detection/llama-13b-concern-detection:latest')
            # artifact_dir = artifact.download()
            # module = DetectConcern()
            # compiled_boostrap = module.load(os.path.join(artifact_dir, f"{model_name}_concerndetect.json"))
            # print("Loaded from artifact")
        # except Exception as e:
            # print(f"Failed to load from artifact due to: {e}")


    evaluate_on_test_set(compiled_boostrap, dev_examples, concern_list)
    evaluate = Evaluate(devset=dev_examples, metric=metric_EM, display_progress=True)
    evaluate(compiled_boostrap)
    return compiled_boostrap, test_examples

def main():
    pipeline, _ = compile_pipeline(model_name="zephyr-7b-beta")


# data = extract_negative_files(os.path.join(ROOT_DIR, "dat", "college", "negative_data", "negative_data_posts_json_"))


if __name__ == "__main__":
    main()

    # see if we can load the compiled pipeline from a json file
    # try:
    #     concern = DetectConcern()
    #     concern.load(os.path.join(ROOT_DIR, "dat", "college", f"{model_name}_concerndetect.json"))
    #     compiled_detect_crises = concern
    #     return compiled_detect_crises, test_examples
    # except Exception as e:
    #     print(f"Failed to load from json file due to: {e}")
