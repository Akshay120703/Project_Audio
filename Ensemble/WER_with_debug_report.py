import numpy as np

def wer_highlight(gt, pred):
    """Compute WER and return highlighted versions of ground truth and predicted output."""
    
    gt_tokens = gt.split()
    pred_tokens = pred.split()
    
    len_gt = len(gt_tokens)
    len_pred = len(pred_tokens)

    dp = np.zeros((len_gt + 1, len_pred + 1), dtype=int)
    operations = np.empty((len_gt + 1, len_pred + 1), dtype=object)

    for i in range(len_gt + 1):
        for j in range(len_pred + 1):
            if i == 0:
                dp[i][j] = j
                operations[i][j] = "I" if j > 0 else None
            elif j == 0:
                dp[i][j] = i
                operations[i][j] = "D" if i > 0 else None
            else:
                if gt_tokens[i - 1] == pred_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                    operations[i][j] = "M"
                else:
                    del_cost = dp[i - 1][j] + 1
                    ins_cost = dp[i][j - 1] + 1
                    sub_cost = dp[i - 1][j - 1] + 1

                    min_cost = min(del_cost, ins_cost, sub_cost)
                    dp[i][j] = min_cost

                    if min_cost == sub_cost:
                        operations[i][j] = f"S{j-1}"
                    elif min_cost == del_cost:
                        operations[i][j] = "D"
                    else:
                        operations[i][j] = f"I{i-1}"

    highlighted_pred = []
    highlighted_gt = []

    i, j = len_gt, len_pred
    while i > 0 or j > 0:
        op = operations[i][j]
        if op == "M":
            highlighted_pred.append(pred_tokens[j - 1])
            highlighted_gt.append(gt_tokens[i - 1])
            i -= 1
            j -= 1
        elif op.startswith("S"):
            highlighted_pred.append(f"[[HIGHLIGHT]]{pred_tokens[j - 1]}[[/HIGHLIGHT]]")
            highlighted_gt.append(f"[[HIGHLIGHT]]{gt_tokens[i - 1]}[[/HIGHLIGHT]]")
            i -= 1
            j -= 1
        elif op == "D":
            highlighted_gt.append(f"[[HIGHLIGHT]]{gt_tokens[i - 1]}[[/HIGHLIGHT]]")
            i -= 1
        elif op.startswith("I"):
            highlighted_pred.append(f"[[HIGHLIGHT]]{pred_tokens[j - 1]}[[/HIGHLIGHT]]")
            j -= 1

    highlighted_pred.reverse()
    highlighted_gt.reverse()

    return " ".join(highlighted_gt), " ".join(highlighted_pred)

def batch_process(input_gt_file, input_pred_file, output_gt_file, output_pred_file):
    """Batch process multiple transcriptions and save highlighted outputs."""
    
    with open(input_gt_file, "r", encoding="utf-8") as gt_f, \
         open(input_pred_file, "r", encoding="utf-8") as pred_f, \
         open(output_gt_file, "w", encoding="utf-8") as out_gt_f, \
         open(output_pred_file, "w", encoding="utf-8") as out_pred_f:
        
        for gt_line, pred_line in zip(gt_f, pred_f):
            gt_line = gt_line.strip()
            pred_line = pred_line.strip()
            
            highlighted_gt, highlighted_pred = wer_highlight(gt_line, pred_line)
            
            out_gt_f.write(highlighted_gt + "\n")
            out_pred_f.write(highlighted_pred + "\n")

if __name__ == "__main__":
    # File paths (update these as needed)
    input_gt_file = "file1.txt"
    input_pred_file = "file2.txt"
    output_gt_file = "highlighted_ground_truth.txt"
    output_pred_file = "highlighted_predicted_output.txt"

    batch_process(input_gt_file, input_pred_file, output_gt_file, output_pred_file)

    print("Batch processing completed! Highlighted files saved.")
