import numpy as np
from Squash_s import squash_arr

# &&&&&&&&&&&&&&&&&&&Consts&&&&&&&&&&&&&&&&&&&&&&&&&&&
# rl_cutoff = 2
N_CLASSSES = 4
layers = 5
std = list()
mean = list()
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def find_min_max_coef(i, j, cls, couple_slice,pred_vect,out_vect):
    '''Inputs:
    i: t (sample in batch); j: l (layer number); cls: s(t) (max class in sample i)
    couple_slice: c(t,l) (Slice the coupling coefficients between layers l and l-1)
    pred_vect: a^ (capsule prediction); out_vect: b (capsule output)
    '''
    global std_mean

    #Sort the coupling coefficients descending
    t = np.argsort(couple_slice)[:][::-1]
    k = 0
    gamma = 0.9
    #A threshold of max capsule output in layer $l$ scaled by gamma$
    thresh = gamma*np.linalg.norm(out_vect[j][i,cls])
    s = 0
    ss = 0
    while (ss < thresh and k < t.size):
        # print("\n couple by prediction",couple_slice[t[k]]*pred_vect[j][i, t[k], cls] )
        s += couple_slice[t[k]]*pred_vect[j][i, t[k], cls] #multiply coefficients by the prediction vector
        ss = np.linalg.norm(squash_arr(s))#squash the vector and compute the norm of the output
        k += 1
    return t[0:k]

def recursive_coupl_coeff(i,j,cls,couple_slice, coupl_coeff,pred_vect,out_vect):
    '''Inputs:
    i: t (sample in batch); j: l (layer number); cls: s(t) (max class in sample i)
    couple_slice: c(t,l) (Slice the coupling coefficients between layers l and l-1)
    coupl_coeff: c (coupling coefficients)
    pred_vect: a^ (capsule prediction); out_vect: b (capsule output)
    '''
    lst_out = list()
    if j == 0 : #layer number
        # Select the maximum coefficients in a reverse order ([::-1] means reverse order)
        return find_min_max_coef(i, j, cls,couple_slice,pred_vect, out_vect)
    else:
        #Select the minimum number of capsules with highest c that result in maximum output val
        max_nodes = find_min_max_coef(i, j, cls,couple_slice,pred_vect,out_vect)
        for xx in max_nodes:
            couple_slice = coupl_coeff[j - 1][i, :, xx]
            lst_out.append(recursive_coupl_coeff(i, j - 1, xx, couple_slice, coupl_coeff, pred_vect, out_vect))
    return lst_out
output=list()
def reemovNestings(l):
    for i in l:
        if ((type(i) == np.ndarray) or (type(i) == list)):
            reemovNestings(i)
        else:
            output.append(i)

def extract_rules_boundary(input_dt, coupl_coeff, pred_vect, out_vect,pred):
    '''Inputs:
    input_dt: x (input vector); coupl_coeff: c (coupling coefficient); pred_vect: a^ (capsule prediction)
     out_vect: b (capsule output); pred: (y^) The output of the capsnet for class prediction in each batch
    '''
    global output
    #Get the maximum class in each sample
    s = np.argmax(pred, axis = 1)
    #Slice the coupling coefficients between two last layers
    couple_slice = coupl_coeff[len(coupl_coeff) - 1]
    #array of rule structure (list by dict), number of lists equal the number of classes
    rule_arr_class = np.empty((N_CLASSSES,), dtype=object)
    # The first set of rules is related to class '0' and the second set of rules is related to class '1'
    for i, v in enumerate(rule_arr_class): rule_arr_class[i] = []
    cnt = 0

    for i in range(len(s)):
        rule_arr = list()
        #Select the coefficients of sample i and maximum class s[i]
        arr_slice = couple_slice[i, :, s[i]]
        j = len(coupl_coeff)-1
        #Recursively find the maximum coefficients in each layer until we reach the input layer
        l = recursive_coupl_coeff(i, j, s[i], arr_slice, coupl_coeff, pred_vect, out_vect)
        #Flatten the list items to have the rules in the format of ((a1 and a2 ..) or b or c)
        flatten_l = list()
        #Each sublist is coming from one node in the last hidden layer (the layer before output)
        # we disjunctively combine all the conditions coming from each last layer node
        for sublist in l:
            reemovNestings(sublist)
            mylist = list(set(output))
            mylist.sort()
            flatten_l.append(mylist)
            output = list()
        for x in flatten_l:
            dict = {}
            # A loop to check whether the new conditions x exist in the previous selected conditions with the same class
            # in the batch
            flg_dict = False
            try:
                for itm in rule_arr_class[s[i]]:
                    lst_temp = list()
                    for key in itm:
                        lst_temp.append(key)
                    #check if x is a sublist of lst_temp or vice versa, in that case we can use absorption law
                    # a and (a or b)=a
                    diff_list = []
                    if (set(x).issubset(lst_temp) or set(lst_temp).issubset(x)):
                        ind = rule_arr_class[s[i]].index(itm)
                        flg_dict = True
                        if (len(x)>len(lst_temp)):
                            diff_list = list(set(x)-set(lst_temp))
                            x = lst_temp
                        else:
                            diff_list = list(set(lst_temp)-set(x))
                        break
            except Exception as e:
                print(e)

            for t in x:
                #if the conditions already exist, add the new condition values to the keys in the corresponding dictionary
                if (flg_dict == True):
                    rule_arr_class[s[i]][ind][t].append(input_dt[i][t])
                #if not, create a new dictionary with the keys
                else:
                    dict.setdefault(t,[]).append(input_dt[i][t])
            #If the new keys are a sublist of any itm in the list and itm has redundant keys, remove all based on
            # absorbtion law
            if((flg_dict == True) and (len(diff_list)>0)):
                for diff_itm in diff_list:
                    del rule_arr_class[s[i]][ind][diff_itm]
            if (flg_dict == False):
                rule_arr_class[s[i]].append(dict)
    #Generate the rules
    rule_lst = list()
    for k in range(N_CLASSSES):
        c = list()
        if(len(rule_arr_class[k])>0):
            c.append('(')
            for itm in rule_arr_class[k]:
                for key in itm:
                    c.append(str(min(itm[key])))
                    c.append('<=')
                    c.append(key)
                    c.append('<=')
                    c.append(str(max(itm[key])))
                    c.append('and')
                del c[-1]
                c.append(') or (')
            del c[-1]
            c.append(')')
            c.append(k)
            rule_lst.append(c)

    return rule_lst
