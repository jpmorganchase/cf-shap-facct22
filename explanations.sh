# Explanations results are separated between close and far for compatibility reasons
python explanations.py --dataset lendingclub --model_version v5 --close 0.5  --results_version v5_close
python explanations.py --dataset lendingclub --model_version v5 --far 0.5  --results_version v5_far

python explanations.py --dataset heloc --model_version v5 --close 0.5 --results_version v5_close
python explanations.py --dataset heloc --model_version v5 --far 0.5 --results_version v5_far

python explanations.py --dataset wines --model_version v5 --close 0.5 --results_version v5_close
python explanations.py --dataset wines --model_version v5 --far 0.5 --results_version v5_far