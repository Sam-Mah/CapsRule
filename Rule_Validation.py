import pandas as pd
from itertools import chain
import sys
from io import StringIO
import numpy as np
from datetime import datetime
from Performance_measures import confusion_metrics_basic, Micro_calculate_measures, Macro_calculate_measures_basic
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# def calculate_measures(tp, tn, fp, fn, uncovered_sample):
#     fn += uncovered_sample
#     try:
#         tpr = float(tp) / (float(tp) + float(fn))
#         accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
#         recall = tpr
#         precision = float(tp) / (float(tp) + float(fp))
#
#         f1_score = (2 * (precision * recall)) / (precision + recall)
#         fp_rate = float(fp) / (float(fp) + float(tn))
#         fn_rate = float(fn) / (float(fn) + float(tp))
#
#         # return precision, recall, f1_score, accuracy, fp_rate, fn_rate
#         PR = str(round(precision * 100, 2))
#         RC = str(round(recall * 100, 2))
#         F1 = str(round(f1_score * 100, 2))
#         ACC = str(round(accuracy * 100, 2))
#         FPR = str(round(fp_rate * 100, 2))
#         FNR = str(round(fn_rate * 100, 2))
#
#         data_pd = [['PR', PR], ['RC', RC], ['F1', F1], ['ACC', ACC], ['FPR', FPR], ['FNR', FNR], ['tp', tp], ['tn', tn],
#                    ['fp', fp], ['fn', fn]]
#
#         df = pd.DataFrame(data_pd, columns=['Measure', 'Percentage'])
#
#     except Exception as e:
#         print(e)
#         data_pd = [['PR', 'Err'], ['RC', 'Err'], ['F1', 'Err'], ['ACC', 'Err'], ['FPR', 'Err'], ['FNR', 'Err']]
#
#         df = pd.DataFrame(data_pd, columns=['Measure', 'Percentage'])
#     return df
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
        # sum_one = 0
        # sum_zero = 0
        sum = np.zeros(N_CLASSSES)
        rule_cnt = 0
        # flg_rule=False
        for rl in rules:
            #Write each rule into a file then execute it
            # f = open('rule_code.py', 'w')
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
                # Generate rule string to be written to a file
                str_rule += ' '.join(rl) + '):\n\t' + 'print(True)\nelse:\n\tprint(False)'
                # f.write(str_rule)
                # f.close()
                '''Save stdout into a file'''
                # Store the reference, in case you want to show things again in standard output
                old_stdout = sys.stdout
                # This variable will store everything that is sent to the standard output
                result = StringIO()
                sys.stdout = result
                # Here we execute the rule instructions, and everything that they will send to standard output will be stored on "result"
                # exec(open("rule_code.py").read())
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
        if((rule_ref[j]/len_val)>0.4):
            del rules[j]
            rule_ref = np.delete(rule_ref, j)
            j-=1
        j+=1
    print(len(rules))
    np.savetxt("Experiments/rules_eval.csv", rules, delimiter=",", fmt='%s')
    return rules
