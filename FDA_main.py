import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from IPython.display import display
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from math import sqrt

# ---------------------------------------------------------------------------------------
# membership function
# ---------------------------------------------------------------------------------------

def linear_equations(lower_limit_range, upper_limit_range, Q1, Q3):
    a1, b1, a2, b2 = 0, lower_limit_range, 0, upper_limit_range

    if lower_limit_range == -1:  # abnormal
        if upper_limit_range == Q3:
            return 0, Q3, 0, 0
        A2 = np.array([[upper_limit_range, 1], [Q3, 1]])
        B2 = np.array([0.5, 1])
        a2, b2 = np.linalg.solve(A2, B2)

        return 0, 0, a2, b2  # normal
    elif lower_limit_range == -2:
        if upper_limit_range == Q1:
            return 0, Q1, 0, 0
        A1 = np.array([[upper_limit_range, 1], [Q1, 1]])
        B1 = np.array([0.5, 1])
        a1, b1 = np.linalg.solve(A1, B1)

        return a1, b1, 0, 0

    if ((lower_limit_range != Q1 and upper_limit_range == Q3) or
            (lower_limit_range != Q1 and upper_limit_range != Q3)):
        A1 = np.array([[lower_limit_range, 1], [Q1, 1]])
        B1 = np.array([0.5, 1])
        a1, b1 = np.linalg.solve(A1, B1)

    if ((lower_limit_range == Q1 and upper_limit_range != Q3) or
            (lower_limit_range != Q1 and upper_limit_range != Q3)):
        A2 = np.array([[upper_limit_range, 1], [Q3, 1]])
        B2 = np.array([0.5, 1])
        a2, b2 = np.linalg.solve(A2, B2)

    return a1, b1, a2, b2


def calculate_membership(data, norm_start, abnorm_start, lower_sus_limit, upper_sus_limit, flag):
    if flag == "PH":
        x_values = np.linspace(6.8, 7.5, 1000)
    elif flag == "Apgar":
        x_values = np.linspace(0, 10, 1000)
    else:
        x_values = np.linspace(-15, 100, 1000)
    data.sort()

    # Suspicious data
    if flag == "BW":
        data_sus = [sus for sus in data if lower_sus_limit < sus < upper_sus_limit]
    else:
        data_sus = [sus for sus in data if lower_sus_limit <= sus <= upper_sus_limit]
    data_sus = np.array(data_sus)

    Q1_sus = np.quantile(data_sus, 0.25)
    Q3_sus = np.quantile(data_sus, 0.75)
    a1_sus, b1_sus, a2_sus, b2_sus = linear_equations(lower_sus_limit, upper_sus_limit, Q1_sus, Q3_sus)

    # Abnormal data
    if flag == "BW":
        data_abn = [abn for abn in data if abn <= abnorm_start]
    else:
        data_abn = [abn for abn in data if abn < abnorm_start]
    data_abn = np.array(data_abn)

    Q1_abn = np.quantile(data_abn, 0.25)
    Q3_abn = np.quantile(data_abn, 0.75)
    _, _, a2_abn, b2_abn = linear_equations(-1, abnorm_start, Q1_abn, Q3_abn)

    # Normal data
    data_norm = np.array([norm for norm in data if norm >= norm_start])

    Q1_norm = np.quantile(data_norm, 0.25)
    Q3_norm = np.quantile(data_norm, 0.75)
    a1_norm, b1_norm, _, _ = linear_equations(-2, norm_start, Q1_norm, Q3_norm)

    if a1_norm != 0 and b1_norm != Q1_norm:
        if flag == "BW":
            y_values_normal = fuzz.trapmf(x_values, [(b1_norm / a1_norm), Q1_norm, 100.0, 100.0])
        else:
            y_values_normal = fuzz.trapmf(x_values,
                                          [abs(b1_norm / a1_norm), Q1_norm, 100.0, 100.0])  # low value, Q3, abs(b2/a2)
    else:
        y_values_normal = fuzz.trapmf(x_values, [Q1_norm, Q1_norm, 100.0, 100.0])

    if a1_sus != 0 and a2_sus != 0:
        y_values_suspicious = fuzz.trapmf(x_values, [abs(b1_sus / a1_sus), Q1_sus, Q3_sus,
                                                     abs(b2_sus / a2_sus)])  # abs(b1/a1), Q1, Q3, abs(b2/a2)
    elif a1_sus != 0 and a2_sus == 0:
        y_values_suspicious = fuzz.trapmf(x_values, [abs(b1_sus / a1_sus), Q1_sus, Q3_sus, Q3_sus])
    elif a1_sus == 0 and a2_sus != 0:
        y_values_suspicious = fuzz.trapmf(x_values, [Q1_sus, Q1_sus, Q3_sus, abs(b2_sus / a2_sus)])
    else:
        y_values_suspicious = fuzz.trapmf(x_values, [Q1_sus, Q1_sus, Q3_sus, Q3_sus])

    if a2_abn != 0 and b2_abn != Q3_abn:
        y_values_abnormal = fuzz.trapmf(x_values, [-20, -20, Q3_abn, abs(b2_abn / a2_abn)])  # low value, Q3, abs(b2/a2)
    else:
        y_values_abnormal = fuzz.trapmf(x_values, [-20, -20, Q3_abn, Q3_abn])

    return y_values_abnormal, y_values_suspicious, y_values_normal, x_values


def plot_results(x_values, y_values_abnormal, y_values_suspicious, y_values_normal, flag):
    plt.figure(figsize=(8, 5))

    plt.plot(x_values, y_values_abnormal, label="Abnormal", linewidth=2, color='red')
    plt.plot(x_values, y_values_suspicious, label="Suspicious", linewidth=2, color='yellow')
    plt.plot(x_values, y_values_normal, label="Normal", linewidth=2, color='green')

    plt.xlabel("x (Fetal Outcome Attribute)")
    plt.ylabel("Membership Degree")
    plt.title("Trapezoidal Membership Functions - " + flag)
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------------------
# rule classification
# ---------------------------------------------------------------------------------------

def rule_classification(rule):
    n_abnormal = rule.count(1.0)
    n_normal = rule.count(-1.0)

    if n_abnormal >= 1:
        return 1.0
    elif n_normal >= 2:
        return -1.0
    else:
        return 0.5


def create_full_rulebase_df():
    statuses = [-1.0, 0.5, 1.0]
    all_rules = list(itertools.product(statuses, repeat=3))

    full_df = pd.DataFrame({
        "Rule": all_rules,
    })
    full_df["Rule-based Classification"] = full_df["Rule"].apply(rule_classification)
    full_df["Max Firing Strength"] = 0.0
    return full_df


def summarize_rules(df):
    full_rule_df = create_full_rulebase_df()
    summary = (
        df.groupby("rule")
        .agg({
            "max firing strength per rule": "first"
        })
        .reset_index()
        .rename(columns={
            "rule": "Rule",
            "max firing strength per rule": "Max Firing Strength"
        })
    )

    full_summary = pd.merge(
        full_rule_df,
        summary,
        on="Rule",
        how="left",
        suffixes=("", "_updated")
    )

    full_summary["Max Firing Strength"] = full_summary["Max Firing Strength_updated"].fillna(
        full_summary["Max Firing Strength"]
    )
    full_summary.drop(columns=["Max Firing Strength_updated"], inplace=True)
    return full_summary


def check_ph_status(value):
    normal_degree = fuzz.interp_membership(x_ph, normal_mf_PH, value)
    sus_degree = fuzz.interp_membership(x_ph, sus_mf_PH, value)
    abnormal_degree = fuzz.interp_membership(x_ph, abnormal_mf_PH, value)
    degrees = {
        -1: normal_degree,
        0.5: sus_degree,
        1: abnormal_degree
    }

    status = max(degrees, key=degrees.get)
    ph_strength = degrees[status]

    return status, ph_strength


def check_bw_status(value):
    normal_degree = fuzz.interp_membership(x_bw, normal_mf_BW, value)
    sus_degree = fuzz.interp_membership(x_bw, sus_mf_BW, value)
    abnormal_degree = fuzz.interp_membership(x_bw, abnormal_mf_BW, value)

    degrees = {
        -1: normal_degree,
        0.5: sus_degree,
        1: abnormal_degree
    }

    status = max(degrees, key=degrees.get)
    BW_strength = degrees[status]

    return status, BW_strength


def check_apgar_status(value):
    normal_degree = fuzz.interp_membership(x_apgar, normal_mf_apgar, value)
    sus_degree = fuzz.interp_membership(x_apgar, sus_mf_apgar, value)
    abnormal_degree = fuzz.interp_membership(x_apgar, abnormal_mf_apgar, value)

    degrees = {
        -1: normal_degree,
        0.5: sus_degree,
        1: abnormal_degree
    }

    status = max(degrees, key=degrees.get)
    agar_strength = degrees[status]

    return status, agar_strength


def evaluate_firing_strength(row):
    return row["Apgar_strength"] * row["PH_strength"] * row["BW_strength"]


def rule_based_classification(row):
    statuses = [row["PH_status"], row["BW_status"], row["Apgar_status"]]
    if 1.0 in statuses:
        return 1
    normal_count = statuses.count(-1.0)
    if normal_count >= 2:
        return -1
    return 0.5


def apply_max_firing_strength(df):
    df["rule"] = list(zip(df["Apgar_status"], df["PH_status"], df["BW_status"]))

    rule_max_strength = df.groupby("rule")["firing_strength"].max().to_dict()

    df["max firing strength per rule"] = df["rule"].map(rule_max_strength)

    return df


def classify_data(df_to_classify):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    df_to_classify[["BW_status", "BW_strength"]] = df_to_classify["Percentile"].apply(check_bw_status).apply(pd.Series)
    df_to_classify[["Apgar_status", "Apgar_strength"]] = df_to_classify["Apgar"].apply(check_apgar_status).apply(
        pd.Series)
    df_to_classify[["PH_status", "PH_strength"]] = df_to_classify["Ph"].apply(check_ph_status).apply(pd.Series)

    df_to_classify["firing_strength"] = df_to_classify.apply(evaluate_firing_strength, axis=1)
    df_to_classify["rule-based classification"] = df_to_classify.apply(rule_based_classification, axis=1)
    df_to_classify = apply_max_firing_strength(df_to_classify)

    summary_df = summarize_rules(df_to_classify)
    # print(df_to_classify)
    df_to_classify = df_to_classify.drop(columns=["BW_status", "BW_strength", "Apgar_status", "Apgar_strength",
                                                  "PH_status", "PH_strength", "matched_rule", "firing_strength"],
                                         errors="ignore")

    return df_to_classify, summary_df


def calculate_TKS_output(summarized_df):
    total_strength = summarized_df["Max Firing Strength"].sum()
    if total_strength > 0:
        weighted_sum = (summarized_df["Rule-based Classification"] * summarized_df["Max Firing Strength"]).sum()
        weighted_average = weighted_sum / total_strength
    else:
        weighted_average = None
    return weighted_average

# ---------------------------------------------------------------------------------------
# grid method
# ---------------------------------------------------------------------------------------
# print(y_val) for each y pred  aand then put the classification

                #  p i delta 25 kombiacji na treining data
                #  potem pomiezy foldami g measure na testowych - wybieramy nAJ
                # WRZUCIC G MEASURE TOP P  I DELTA I PRZETESTOWAC ZNOW NA TEST DATA

##### GRID na podstawie Ewy kodu
def grid_method_evaluate_firing_strength(row):
    return row["Apgar_strength"] * row["PH_strength"] * row["BW_strength"]


def grid_method_rule_based_classification(row, p_value=0.0):
    statuses = [row["PH_status"], row["BW_status"], row["Apgar_status"]]

    if 1.0 in statuses:
        return 1.0
    normal_count = statuses.count(-1.0)
    if normal_count >= 2:
        return -1.0
    else:
        return p_value

#for each 27 rules
def apply_max_firing_strength(df):
    df["rule"] = list(zip(df["Apgar_status"], df["PH_status"], df["BW_status"]))
    rule_max_strength = df.groupby("rule")["firing_strength"].max().to_dict()
    df["max firing strength per rule"] = df["rule"].map(rule_max_strength)
    return df

#classify data with p value
#wrzucamy tu nasze mnozenie zamiast min
def _grid_method_classify_data(df_to_classify, p_value=0.0):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    df_to_classify[["BW_status", "BW_strength"]] = df_to_classify["Percentile"].apply(check_bw_status).apply(pd.Series)
    df_to_classify[["Apgar_status", "Apgar_strength"]] = df_to_classify["Apgar"].apply(check_apgar_status).apply(
        pd.Series)
    df_to_classify[["PH_status", "PH_strength"]] = df_to_classify["Ph"].apply(check_ph_status).apply(pd.Series)

    df_to_classify["firing_strength"] = df_to_classify.apply(grid_method_evaluate_firing_strength, axis=1)

    df_to_classify["rule-based classification"] = df_to_classify.apply(
        lambda row: grid_method_rule_based_classification(row, p_value), axis=1
    )

    df_to_classify = apply_max_firing_strength(df_to_classify)

    summary_df = summarize_rules(df_to_classify)

    df_to_classify = df_to_classify.drop(columns=["BW_status", "BW_strength", "Apgar_status", "Apgar_strength",
                                                  "PH_status", "PH_strength", "matched_rule", "firing_strength"],
                                         errors="ignore")
    return df_to_classify, summary_df

def grid_method_calculate_TKS_output(summarized_df):
    total_strength = summarized_df["Max Firing Strength"].sum()

    if total_strength > 0:
        weighted_sum = (summarized_df["Rule-based Classification"] * summarized_df["Max Firing Strength"]).sum()
        weighted_average = weighted_sum / total_strength
    else:
        weighted_average = 0.0
    return weighted_average

def g_measure(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    tn, fp, fn, tp = conf_matrix.ravel()
    print(conf_matrix)
    if (tp + fn) == 0:
        sensitivity = 0.0
    else:
        sensitivity = tp / (tp + fn)

    if (tn + fp) == 0:
        specificity = 0.0
    else:
        specificity = tn / (tn + fp)

    if sensitivity == 0 or specificity == 0:
        return 0.0

    return sqrt(sensitivity * specificity)

#trzeba bylo zmienic wdg tego co mowila jagoda - inacze bylby information leakage bo dzialamy na tych samych danych dla p i delta
#  5-fold cross-validation:
# in each iteratuon we need to have
#   - 4 parts (80%) ➜ trening + searching for the best p and delta
#   - 1 part (20%) ➜ test -> only for testing the g measure with the parameters p delta
# we do it 5 times each time on a different test data
# mean G-measure for test data from each fold - the final optimum !!!!!!!!!!!! - mediana

def grid_search(df, p_values, delta_values):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_g_measures = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        print(f"\n Fold {fold + 1} ")
        train_df = df.iloc[train_idx].copy() #pdozial train test
        test_df = df.iloc[test_idx].copy()

        best_g_train = 0.0
        best_params_train = (None, None)

        # Grid search on the train data
        for p in p_values:
            for delta in delta_values:
                y_pred_train = []
                for idx, row in train_df.iterrows():
                    sample = train_df.loc[[idx]]
                    _, sample_summary = _grid_method_classify_data(sample, p_value=p)
                    y0 = grid_method_calculate_TKS_output(sample_summary)
                    if y0 > delta:
                        y_pred_train.append(1)
                    elif y0 < -delta:
                        y_pred_train.append(-1)
                    else:
                        y_pred_train.append(0)

                y_true_train = train_df["E3"].values
                g_train = g_measure(y_true_train, y_pred_train)
                print(f"Params: p = {p}, delta = {delta};  G-measure = {g_train:.4f}")

                if g_train > best_g_train:
                    best_g_train = g_train
                    best_params_train = (p, delta)

        print(f"Best from fold {fold + 1} → G-measure = {best_g_train:.4f}, p_value = {best_params_train[0]}, delta = {best_params_train[1]}")

        # Test data with the best params
        y_pred_test = []
        for idx, row in test_df.iterrows():
            sample = test_df.loc[[idx]]
            _, sample_summary = _grid_method_classify_data(sample, p_value=best_params_train[0])
            y0 = grid_method_calculate_TKS_output(sample_summary)
            if y0 > best_params_train[1]: #delta
                y_pred_test.append(1)
            elif y0 < -best_params_train[1]:
                y_pred_test.append(-1)
            else:
                y_pred_test.append(0)

        y_true_test = test_df["E3"].values
        g_test = g_measure(y_true_test, y_pred_test)
        print(f"G-measure on the test data = {g_test:.4f}")
        all_g_measures.append(g_test)

    final_median_g = np.median(all_g_measures)
    print(f"\nResults across all folds: Median G-measure = {final_median_g:.4f}") # delta
    return final_median_g


# ---------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    file_path = "FDA-data-e.xls"
    data = pd.read_excel(file_path)
    expert_cols = ['E3', 'Ph', 'Apgar', 'Percentile']
    data_to_classify = data[expert_cols].copy()

    # data_to_classify = data[["Percentile", "Apgar", "Ph"]].copy()
    percentile = data.iloc[:, 0].to_numpy().copy()
    Apgar = data.iloc[:, 1].to_numpy().copy()
    Ph = data.iloc[:, 2].to_numpy().copy()

    abnormal_mf_PH, sus_mf_PH, normal_mf_PH, x_ph = calculate_membership(Ph, 7.2, 7.1, 7.1, 7.2, "PH")
    abnormal_mf_apgar, sus_mf_apgar, normal_mf_apgar, x_apgar = calculate_membership(Apgar, 7, 5, 5, 6.99, "Apgar")
    abnormal_mf_BW, sus_mf_BW, normal_mf_BW, x_bw = calculate_membership(percentile, 10, 5, 5, 10, "BW")

    # plot_results(x_ph, abnormal_mf_PH, sus_mf_PH, normal_mf_PH, "PH")
    # plot_results(x_bw, abnormal_mf_BW, sus_mf_BW, normal_mf_BW, "BW")
    # plot_results(x_apgar, abnormal_mf_apgar, sus_mf_apgar, normal_mf_apgar, "Apgar")

    classified_data, summarized_df = classify_data(data_to_classify)
    summarized_df = summarized_df.set_index("Rule")
    TKS_output = calculate_TKS_output(summarized_df)

    p_values = [-0.5, -0.25, 0, 0.25, 0.5]
    delta_values = [-0.5, -0.25, 0, 0.25, 0.5]

    print("\nGrid search method : ")
    final_g = grid_search(data_to_classify, p_values, delta_values)
    print(f"\nFinal G - measure output (median) : {final_g:.4f}")