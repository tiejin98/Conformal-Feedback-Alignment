import numpy as np

acc_list = []
rel_list = []
comp_list = []
expr_list = []
overall_list = []
a = np.load("evaluation_scores_CFA.pkl",allow_pickle=True)
print(a)
key_list = list(a.keys())[:]
for key in key_list:
    try:
        acc = a[key]["Acc"]
        rel = a[key]["Rel"]
        comp = a[key]["Comp"]
        expr = a[key]["Expr"]

        acc_list.append(acc)
        rel_list.append(rel)
        comp_list.append(comp)
        expr_list.append(expr)
        overall_list.append(np.mean([acc, rel, comp, expr]))
    except:
        continue
print(len(a))
print("Acc: ", np.mean(acc_list))
print("Rel: ", np.mean(rel_list))
print("Comp: ", np.mean(comp_list))
print("Expr: ", np.mean(expr_list))
print("Overall: ", np.mean(overall_list))

print("std:", np.var(overall_list))

