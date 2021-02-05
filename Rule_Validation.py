import pandas as pd
from itertools import chain
import sys
from io import StringIO
import numpy as np
from datetime import datetime
from Performance_measures import confusion_metrics_basic, Micro_calculate_measures, Macro_calculate_measures_basic
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
N_CLASSSES = 4
def validate_rules(rule_set, x_val, y_val_one):
    # Flatten the rule_set list into a 1D list
    # print(rule_set)
    rules = list(chain.from_iterable(rule_set))
    len_rules = len(rules)
    len_val = len(y_val_one)
    y_val = [np.where(r == 1)[0][0] for r in y_val_one]
    y_pred = np.zeros(len(y_val), dtype=int)
    rule_ref = np.zeros(len_rules, dtype=int)

    t = 0
    uncovered_sample = 0
    print("Today's date:", datetime.now())
    for x in x_val:
        # break
        flg = False
        sum = np.zeros(N_CLASSSES)
        rule_cnt = 0
        for rl in rules:
            str_rule = 'if ('
            cnt_non_match = 0
            rl_lbl = rl[-1]
            rl = rl[:-1]
            i = 3
            while i < (len(rl) - 1):
                try:
                    rl[i] = str(x[int(rl[i])])
                    i += 6
                except Exception as E:
                    print(E)
            try:
                # Generate rule string
                str_rule += ' '.join(rl) + '):\n\t' + 'print(True)\nelse:\n\tprint(False)'
                # Store the reference, in case you want to show things again in standard output
                old_stdout = sys.stdout
                # This variable will store everything that is sent to the standard output
                result = StringIO()
                sys.stdout = result
                # Here we execute the rule instructions, and everything that they will send to standard output will be stored on "result"
                exec(str_rule)
                # Redirect again the std output to screen
                sys.stdout = old_stdout
                # Then, get the stdout like a string and process it!
                flg_rule = result.getvalue()
                flg_rule = flg_rule.replace('\n', '')

            except Exception as E:
                print(E)
            # If the rule covers the whole sample
            if (flg_rule == 'True'):
                # Check the strength of the rule by counting the number of the times the rule mis-predicts a sample
                flg = True
                sum[int(rl_lbl)] += 1
                if (rl_lbl != y_val[t]):
                    rule_ref[rule_cnt]+=1
            rule_cnt += 1
        # if the current sample is not covered by any of the rules
        if (flg == False):
            uncovered_sample += 1
            continue
        # if the current sample is covered by at least one of the rules
        else:
            rule_vote = np.argmax(sum)

        ## In the rules, output 0 means (1 0) Malicious
        # In the rules, output 1 means (0 1) Benign
        y_pred[t] = rule_vote
        t+=1

    mcm, tp_mean, tn_mean, fp_mean, fn_mean = confusion_metrics_basic(y_val, y_pred)
    out_measures = Micro_calculate_measures(tp_mean, tn_mean, fp_mean, fn_mean, uncovered_sample)
    pr, rc, f1 = Macro_calculate_measures_basic(y_val, y_pred)
    np.savetxt("Experiments/Micro_Validation_Conf_CapsRule.csv", mcm, delimiter=',', fmt='%s')
    np.savetxt("Experiments/Micro_Validation_Measures_CapsRules.csv", out_measures.to_numpy(), delimiter=',', fmt='%s')
    f = open("Experiments/Macro_Validation_Results_CapsRules.txt", 'w')
    f_str = "PR:" + str(pr) + " RC:" + str(rc) + " F1:" + str(f1)
    f.write(f_str)
    f.close()
    np.savetxt("Experiments/y_true_CapsRule_Val.csv", y_val, delimiter=',')
    np.savetxt("Experiments/y_pre_CapsRule_Val.csv", y_pred, delimiter=',')

    j = 0
    print("Today's date:", datetime.now())
    print(len(rules))
    np.savetxt("Experiments/rules_val.csv", rules, delimiter=",", fmt='%s')
    while (j<len(rule_ref)):
        #delta=0.4
        if((rule_ref[j]/len_val)>0.4):
            del rules[j]
            rule_ref = np.delete(rule_ref, j)
            j-=1
        j+=1
    print(len(rules))
    np.savetxt("Experiments/rules_eval.csv", rules, delimiter=",", fmt='%s')
    return rules
