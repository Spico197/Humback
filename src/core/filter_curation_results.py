import argparse
import re
from collections import Counter

from tqdm import tqdm

from src.utils.io import dump_jsonlines, load_jsonlines


def main(args):
    data = load_jsonlines(args.data_filepath)
    regex = re.compile(r"[Ss]core:\s*(\d+)")

    scores = []
    qualified_results = []
    all_results = []
    for ins in tqdm(data, desc="Filtering curation results"):
        raw = ins["raw"]
        curation_response = ins[args.curation_response_column_name]
        score_matched = regex.search(curation_response)
        if score_matched:
            score = int(score_matched.group(1))
        else:
            score = None

        scores.append(str(score))

        all_results.append(
            {
                "instruction": raw["generated_instruction"],
                "response": raw["response"],
                "score": score,
            }
        )
        if isinstance(score, int) and score is not None:
            if score >= args.min_score:
                qualified_results.append(
                    {
                        "instruction": raw["generated_instruction"],
                        "response": raw["response"],
                        "score": score,
                    }
                )

    dump_jsonlines(all_results, args.middle_save_filepath)
    dump_jsonlines(qualified_results, args.save_filepath)

    print(Counter(scores).most_common())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filepath", type=str)
    parser.add_argument("--middle_save_filepath", type=str)
    parser.add_argument("--save_filepath", type=str)
    parser.add_argument("--curation_response_column_name", type=str, default="response")
    parser.add_argument("--min_score", type=int, default=5)
    args = parser.parse_args()

    main(args)
