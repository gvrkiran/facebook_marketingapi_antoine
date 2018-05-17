import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import glob
import numpy as np
import random
from matplotlib.patches import Rectangle

TOP_PERC = 50


#############
# FUNCTIONS #
#############

def global_score(target_file, dest_file, home_file, score_type_, nb_int):
    """
    This function computes the assimilation score for a given target populations coming from some home population and
    trying to assimilate to a certain dest population
    :param target_file: File containing interests audiences for the target population
    :param dest_file: File containing interests audiences for the dest population
    :param home_file: File containing interests audiences for the home population
    :param score_type_: String indicating if the score should be computed using subtraction or division
    :param nb_int: Number of interests to consider
    :return: scores: the per-interest assimilation scores for each most german interests
             nb_target: the size of the target population
    """
    target_data = pd.read_csv(target_file, index_col=0)
    dest_data = pd.read_csv(dest_file, index_col=0)
    home_data = pd.read_csv(home_file, index_col=0)

    # Remove hand-picked interests
    target_audience = target_data['audience'][0:3000]
    dest_audience = dest_data['audience'][0:3000]
    home_audience = home_data['audience'][0:3000]

    nb_target = target_audience[0]
    nb_dest = dest_audience[0]
    nb_home = home_audience[0]

    # Remove erroneous audiences
    target_errors = (target_audience != nb_target)
    dest_errors = (dest_audience != nb_dest)
    home_errors = (home_audience != nb_home)
    errors = target_errors | dest_errors | home_errors
    target_audience = target_audience[errors]
    dest_audience = dest_audience[errors]
    home_audience = home_audience[errors]

    # Select a certain number of interests
    random.seed(0)
    int_ind = random.sample(list(dest_audience.index), nb_int)
    int_ind = np.sort(int_ind)

    target_audience = target_audience[int_ind]
    dest_audience = dest_audience[int_ind]
    home_audience = home_audience[int_ind]

    # Compute activity level
    target_nb_interests = target_audience.shape[0]
    total_nb_interested_target = target_audience.sum(0)
    dest_nb_interests = dest_audience.shape[0]
    total_nb_interested_dest = dest_audience.sum(0)
    home_nb_interests = home_audience.shape[0]
    total_nb_interested_home = home_audience.sum(0)

    # Compute interest ratios
    target_ir = target_audience.values / float(total_nb_interested_target)
    dest_ir = dest_audience.values / float(total_nb_interested_dest)
    home_ir = home_audience.values / float(total_nb_interested_home)

    # Keep only 'dest' interests
    dest_indexes = dest_ir > home_ir
    g_dest_ir = dest_ir[dest_indexes]
    g_home_ir = home_ir[dest_indexes]
    g_target_ir = target_ir[dest_indexes]

    # Keep only 'very dest' interests
    if score_type_ == '-':
        dest_home_perc = np.percentile(g_dest_ir - g_home_ir, TOP_PERC)
        very_dest_indexes = (g_dest_ir - g_home_ir) > dest_home_perc
    else:
        dest_home_perc = np.percentile(g_dest_ir / g_home_ir, TOP_PERC)
        very_dest_indexes = ((g_dest_ir / g_home_ir) > dest_home_perc)

    vg_dest_ir = g_dest_ir[very_dest_indexes]
    vg_target_ir = g_target_ir[very_dest_indexes]

    # Compute scores
    if score_type_ == '-':
        scores = vg_target_ir - vg_dest_ir
    else:
        scores = vg_target_ir / vg_dest_ir

    return scores, nb_target


# Get score from the individual states file, allows to compute new scores
def state_specific_scores(target_dir, dest_dir_or_file, home_file, score_type_, nb_int):
    """
    This function computes the assimilation score for a given target populations coming from some home population and
    trying to assimilate to a certain dest population.
    In this case, the assimilation score is computed for different regions of the destination indiviually.
    :param target_dir: Directory of files containing interests audiences for the target population in each region
    :param dest_dir_or_file: Directory/File containing interests audiences for the dest population
                             in each region/for the whole country
    :param home_file: File containing interests audiences for the home population
    :param score_type_: String indicating if the score should be computed using subtraction or division
    :param nb_int: Number of interests to consider
    :return: scores: the per-interest assimilation scores for each most german interests for each region
             target_pop_sizes: the size of the target population in each region
    """
    target_files = glob.glob(target_dir + '*.csv')
    dest_files = glob.glob(dest_dir_or_file + "*.csv")

    global_dest = False
    if dest_files == 1:
        global_dest = True

    scores = []
    target_pop_sizes_ = []
    for j in range(len(target_files)):

        target_data = pd.read_csv(target_files[j], index_col=0)
        if global_dest:
            dest_data = pd.read_csv(dest_files[j], index_col=0)
        else:
            dest_data = pd.read_csv(dest_files[0], index_col=0)
        home_data = pd.read_csv(home_file, index_col=0)

        target_audience = target_data['audience'][0:3000]
        dest_audience = dest_data['audience'][0:3000]
        home_audience = home_data['audience'][0:3000]

        random.seed(10)
        int_ind = random.sample(range(1, target_audience.shape[0] - 1), nb_int)
        int_ind.insert(0, 0)
        int_ind = np.sort(int_ind)

        target_audience = target_audience[int_ind]
        dest_audience = dest_audience[int_ind]
        home_audience = home_audience[int_ind]

        nb_target = target_data['audience'][0]
        nb_dest = dest_data['audience'][0]
        nb_home = home_data['audience'][0]

        # Remove erroneous audiences
        target_errors = (target_audience != nb_target)
        dest_errors = (dest_audience != nb_dest)
        home_errors = (home_audience != nb_home)
        errors = target_errors | dest_errors | home_errors
        target_audience = target_audience[errors]
        dest_audience = dest_audience[errors]
        home_audience = home_audience[errors]

        # Compute activity level
        target_nb_interests = target_audience.shape[0]
        total_nb_interested_target = target_audience.sum(0)
        dest_nb_interests = dest_audience.shape[0]
        total_nb_interested_dest = dest_audience.sum(0)
        home_nb_interests = home_audience.shape[0]
        total_nb_interested_home = home_audience.sum(0)

        # Compute interest ratios
        target_ir = target_audience.values / float(total_nb_interested_target)
        dest_ir = dest_audience.values / float(total_nb_interested_dest)
        home_ir = home_audience.values / float(total_nb_interested_home)

        # Keep only 'dest' interests
        dest_indexes = dest_ir > home_ir
        g_target_ir = target_ir[dest_indexes]
        g_dest_ir = dest_ir[dest_indexes]
        g_home_ir = home_ir[dest_indexes]

        # Keep only 'very dest' interests
        if score_type_ == '-':
            dest_home_perc = np.percentile(g_dest_ir - g_home_ir, TOP_PERC)
            very_dest_indexes = (g_dest_ir - g_home_ir) > dest_home_perc
        else:
            dest_home_perc = np.percentile(g_dest_ir / g_home_ir, TOP_PERC)
            very_dest_indexes = ((g_dest_ir / g_home_ir) > dest_home_perc)
        vg_dest_ir = g_dest_ir[very_dest_indexes]
        vg_target_ir = g_target_ir[very_dest_indexes]

        # Compute scores
        if score_type_ == '-':
            scores.append(vg_target_ir - vg_dest_ir)
        else:
            scores.append(vg_target_ir / vg_dest_ir)

        target_pop_sizes_.append(nb_target)

    return scores, target_pop_sizes_


def get_top_bottom_interests(dest_file, home_file, dest_name, home_name):
    """
    This function computes the most and least distinct interest for a certain dest population compared to a certain home
    population.
    :param dest_file: File containing interests audiences for the dest population
    :param home_file: File containing interests audiences for the home population
    :param dest_name: Identifier for the dest population
    :param home_name: Identifier for the home population
    """
    dest_data = pd.read_csv(dest_file, index_col=0)
    home_data = pd.read_csv(home_file, index_col=0)

    nb_dest = dest_data['audience'][0]
    nb_home = home_data['audience'][0]

    # Remove erroneous audiences
    dest_errors = (dest_data['audience'] != nb_dest)
    home_errors = (home_data['audience'] != nb_home)
    errors = dest_errors | home_errors
    dest_data = dest_data[errors]
    home_data = home_data[errors]

    # Compute activity level
    dest_nb_interests = dest_data.shape[0]
    total_nb_interested_dest = dest_data["audience"].sum(0)
    home_nb_interests = home_data.shape[0]
    total_nb_interested_home = home_data['audience'].sum(0)

    # Compute interest ratios
    dest_data['CIR'] = dest_data['audience'] / float(total_nb_interested_dest)
    home_data['CIR'] = home_data['audience'] / float(total_nb_interested_home)
    dest_data['CIR diff'] = dest_data['CIR'] - home_data['CIR']
    data_diff = dest_data.sort_values('CIR diff')
    dest_data['CIR div'] = dest_data['CIR'] / home_data['CIR']
    data_div = dest_data.sort_values('CIR div')

    least_dest_diff = data_diff.head(100)
    most_dest_diff = data_diff.tail(100)
    most_dest_diff = most_dest_diff.iloc[::-1]
    least_dest_div = data_div.head(100)
    most_dest_div = data_div.tail(100)
    most_dest_div = most_dest_div.iloc[::-1]

    # least_dest_diff['name'].to_csv("list_least_" + dest_name + "interest_compared_to_" + home_name + '_diff.csv')
    # most_dest_diff['name'].to_csv("list_most_" + dest_name + "interest_compared_to_" + home_name + '_diff.csv')
    least_dest_div['name'].to_csv("list_least_" + dest_name + "_interests_compared_to_" + home_name + '_div.csv')
    most_dest_div['name'].to_csv("list_most_" + dest_name + "_interests_compared_to_" + home_name + '_div.csv')

def combine_sub_pop(sub_pops_fns, combined_pop_fn):
    '''
    This function combine sub-populations audience according to one criteria (e.g. age, gender, education, ...) by
    adding them together.
    :param sub_pops_fns: List the names of the file containing the audiences for the sub-populations
    :param combined_pop_fn: Name of the file containing the combined audiences
    '''

    combined_data = pd.read_csv(sub_pops_fns[0], index_col=0)

    for file in sub_pops_fns[1:]:
        data = pd.read_csv(file, index_col=0)
        combined_data['Target in Germany audience'] += data['Target in Germany audience']

    combined_data.to_csv(combined_pop_fn)


##########
# SCRIPT #
##########

score_type = '/'
states_name = ["Baden-Wurttemberg", "Bayern", "Berlin", "Brandenburg", "Bremen", "Hamburg",
               "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz",
               "Saarland", "Sachsen", "Saxony-Anhalt", "Schleswig-Holstein", "Thuringen"]


###################################
# Most and Least German interests #
###################################

get_top_bottom_interests("interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
                         "interests_audiences/20180110-155243_global_3088_interests_nonexp_ES.csv",
                         "DE", "ES")
if False:


    #############################
    # Combining sub populations #
    #############################
    '''
    combine_sub_pop(["interests_audiences/20180115-223022_global_3088_interests_Arab_17_DE.csv",
                    "interests_audiences/20180115-225302_global_3088_interests_Arab_18_24_DE.csv",
                    "interests_audiences/20180115-233139_global_3088_interests_Arab_25_44_DE.csv",
                    "interests_audiences/20180115-235302_global_3088_interests_Arab_45_64_DE.csv",
                    "interests_audiences/20180116-083637_global_3088_interests_Arab_65_DE.csv"],
                    "interests_audiences/global_3088_interests_Arab_age_sum_DE.csv")
    '''

    ###############################################
    # Assimilation score for non-Arab populations #
    ###############################################

    x_low = 0.4
    x_high = 1
    y_low = 0
    y_high = 1.05
    matplotlib.rcParams.update({'font.size': 18})

    target_pop_sizes = np.zeros(6)
    target_activity_levels = np.zeros(6)
    pops = ['arab', 'turkish', 'turkish_nonexp', 'spanish', 'french', 'austrian']
    pops_c = ['Arabic Sp. Ex.', 'Turkish Sp.', 'Turkish Sp. Non-Ex.', 'Spanish Ex.', 'French Ex.', 'Austrian Ex.']
    col = ['k', 'g', 'b', 'm', 'c', 'r']

    # Variation by interest number
    nb_ints = range(7, 2908, 100)
    medians_arab = []
    medians_turkisk = []
    medians_turkisk_nonexp = []
    medians_spanish = []
    medians_french = []
    medians_austrian = []
    for nb_int in nb_ints:
        score_0, target_pop_sizes[0] = global_score(
            "interests_audiences/20171229-124714_global_3088_interests_Arab_sp_exp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        score_1, target_pop_sizes[1] = global_score(
            "interests_audiences/20171229-122014_global_3088_interests_Turkish_sp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180110-165817_global_3088_interests_nonexp_TR.csv",
            score_type, nb_int)
        score_2, target_pop_sizes[2] = global_score(
            "interests_audiences/20171228-090414_global_3088_interests_Turkish_sp_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180110-165817_global_3088_interests_nonexp_TR.csv",
            score_type, nb_int)
        score_3, target_pop_sizes[3] = global_score(
            "interests_audiences/20171229-140100_global_3088_interests_Spanish_exp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180110-155243_global_3088_interests_nonexp_ES.csv",
            score_type, nb_int)
        score_4, target_pop_sizes[4] = global_score(
            "interests_audiences/20171229-131234_global_3088_interests_French_exp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180110-144143_global_3088_interests_nonexp_FR.csv",
            score_type, nb_int)
        score_5, target_pop_sizes[5] = global_score(
            "interests_audiences/20171229-133929_global_3088_interests_Austrian_exp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180110-151539_global_3088_interests_nonexp_AT.csv",
            score_type, nb_int)

        medians_arab.append(np.median(score_0))
        medians_turkisk.append(np.median(score_1))
        medians_turkisk_nonexp.append(np.median(score_2))
        medians_spanish.append(np.median(score_3))
        medians_french.append(np.median(score_4))
        medians_austrian.append(np.median(score_5))

    # Compute the median of the per-interest assimilation score
    print("Median score for Arab speaking ex-pats: ", np.median(score_0))
    print("Median score for Turkish speaking: ",np.median(score_1))
    print("Median score for Turkish speaking ex-pats: ",np.median(score_2))
    print("Median score for Spanish ex-pats: ",np.median(score_3))
    print("Median score for French ex-pats: ",np.median(score_4))
    print("Median score for Austrian ex-pats: ",np.median(score_5))

    # Compute the max deviation
    maximums = list()
    maximums.append(np.max(abs(medians_arab[5:]-np.mean(medians_arab))/np.mean(medians_arab)))
    maximums.append(np.max(abs(medians_turkisk[5:]-np.mean(medians_turkisk))/np.mean(medians_turkisk)))
    maximums.append(np.max(abs(medians_turkisk_nonexp[5:]-np.mean(medians_turkisk_nonexp))/np.mean(medians_turkisk_nonexp)))
    maximums.append(np.max(abs(medians_spanish[5:]-np.mean(medians_spanish))/np.mean(medians_spanish)))
    maximums.append(np.max(abs(medians_french[5:]-np.mean(medians_french))/np.mean(medians_french)))
    maximums.append(np.max(abs(medians_austrian[5:]-np.mean(medians_austrian))/np.mean(medians_austrian)))
    # print("Max of Maxs", np.max(maximums))
    # print("Mean of max", np.mean(maximums))

    # Plotting the evolution of the medians of the assimilation scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.plot(nb_ints, medians_arab, c=col[0])
    rects2 = ax.plot(nb_ints, medians_turkisk, c=col[1])
    rects3 = ax.plot(nb_ints, medians_turkisk_nonexp, c=col[2])
    rects4 = ax.plot(nb_ints, medians_spanish, c=col[3])
    rects5 = ax.plot(nb_ints, medians_french, c=col[4])
    rects6 = ax.plot(nb_ints, medians_austrian, c=col[5])
    ax.set_ylabel("Assimilation score")
    ax.legend((rects6[0], rects5[0], rects4[0], rects3[0], rects2[0], rects1[0]),
               ("Target = " + pops_c[5], "Target = " + pops_c[4], "Target = " + pops_c[3],
                "Target = " + pops_c[2], "Target = " + pops_c[1], "Target = " + pops_c[0]), loc="lower right", prop={'size': 10})
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel("Number of interests")

    plt.savefig("final_diverse_pops_evolution.pdf", format='pdf', bbox_inches='tight')


    # Plotting the median of the assimilation scores for the largest number of interests
    fig = plt.figure()
    ind = np.arange(6)  # the y locations for the groups
    height = 0.5      # the width of the bars
    ax = fig.add_subplot(111)
    rects2 = ax.barh(ind, [np.median(score_0), np.median(score_1), np.median(score_2), np.median(score_3),
                            np.median(score_4), np.median(score_5)], height)
    rects2[0].set_color(col[0])
    rects2[1].set_color(col[1])
    rects2[2].set_color(col[2])
    rects2[3].set_color(col[3])
    rects2[4].set_color(col[4])
    rects2[5].set_color(col[5])
    rects2[5].set_hatch("x")
    ax.set_yticks(ind)
    ax.set_xlabel("Assimilation score")
    ax.set_yticklabels(pops_c)
    ax.set_xlim(x_low, x_high)

    plt.savefig("final_diverse_pops.pdf", format='pdf', bbox_inches='tight')


    ##########################################
    # Assimilation score for Arab sub-groups #
    ##########################################

    arab_target_pop_sizes = np.zeros(16)
    arab_pops = ['arab', 'arab_men', 'arab_women', 'arab_uni', 'arab_nonuni']
    arab_pops_c = ['All', 'Men', 'Women', 'University Graduate', 'Not University Graduate', '<18', '18-24', '25-44', '45-64', '>64']

    # Variation by interest number
    medians_arab_gender = []
    medians_arab_uni = []
    medians_arab_age = []
    medians_men = []
    medians_women = []
    medians_uni = []
    medians_nonuni = []
    medians_17 = []
    medians_24 = []
    medians_44 = []
    medians_64 = []
    medians_65 = []
    medians_men_24_uni = []
    medians_men_44_uni = []
    medians_women_64_nonuni = []
    for nb_int in nb_ints:

        arab_score_0_gender, arab_target_pop_sizes[0] = global_score(
            "interests_audiences/20171229-153239_global_3088_interests_Arab_sp_exp_gender_sum_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_0_uni, arab_target_pop_sizes[1] = global_score(
            "interests_audiences/20180101-154345_global_3088_interests_Arab_sp_exp_education_sum_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_0_age, arab_target_pop_sizes[2] = global_score(
            "interests_audiences/20180115-223022_global_3088_interests_Arab_sp_exp_age_sum_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_1, arab_target_pop_sizes[3] = global_score(
            "interests_audiences/20171229-153239_global_3088_interests_Arab_sp_exp_men_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_2, arab_target_pop_sizes[4] = global_score(
            "interests_audiences/20171230-200302_global_3088_interests_Arab_sp_exp_women_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_3, arab_target_pop_sizes[5] = global_score(
            "interests_audiences/20180101-154345_global_3088_interests_Arab_sp_exp_uni_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_4, arab_target_pop_sizes[6] = global_score(
            "interests_audiences/20180101-213231_global_3088_interests_Arab_sp_exp_nonuni_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_5, arab_target_pop_sizes[7] = global_score(
            "interests_audiences/20180115-223022_global_3088_interests_Arab_sp_exp_17_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_6, arab_target_pop_sizes[8] = global_score(
            "interests_audiences/20180115-225302_global_3088_interests_Arab_sp_exp_18_24_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_7, arab_target_pop_sizes[9] = global_score(
            "interests_audiences/20180115-233139_global_3088_interests_Arab_sp_exp_25_44_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_8, arab_target_pop_sizes[10] = global_score(
            "interests_audiences/20180115-235302_global_3088_interests_Arab_sp_exp_45_64_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_9, arab_target_pop_sizes[11] = global_score(
            "interests_audiences/20180116-083637_global_3088_interests_Arab_sp_exp_65_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_10, arab_target_pop_sizes[12] = global_score(
            "interests_audiences/20180123-095352_global_3088_interests_Arab_sp_exp_18_24_men_uni_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_11, arab_target_pop_sizes[13] = global_score(
            "interests_audiences/20180123-101748_global_3088_interests_Arab_sp_exp_25_44_men_uni_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_12, arab_target_pop_sizes[14] = global_score(
            "interests_audiences/20180123-111302_global_3088_interests_Arab_sp_exp_45_64_women_nonuni_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)
        arab_score_13, arab_target_pop_sizes[15] = global_score(
            "interests_audiences/20180124-095604_global_3088_interests_Arab_sp_exp_bundesliga_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE.csv",
            "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
            score_type, nb_int)

        medians_arab_gender.append(np.median(arab_score_0_gender))
        medians_arab_uni.append(np.median(arab_score_0_uni))
        medians_arab_age.append(np.median(arab_score_0_age))
        medians_men.append(np.median(arab_score_1))
        medians_women.append(np.median(arab_score_2))
        medians_uni.append(np.median(arab_score_3))
        medians_nonuni.append(np.median(arab_score_4))
        medians_17.append(np.median(arab_score_5))
        medians_24.append(np.median(arab_score_6))
        medians_44.append(np.median(arab_score_7))
        medians_64.append(np.median(arab_score_8))
        medians_65.append(np.median(arab_score_9))
        medians_men_24_uni.append(np.median(arab_score_10))
        medians_men_44_uni.append(np.median(arab_score_11))
        medians_women_64_nonuni.append(np.median(arab_score_12))


    print("Median score for men: ", medians_men[-1])
    print("Median score for women: ", medians_women[-1])
    print("Median score for uni: ", medians_uni[-1])
    print("Median score for nonuni: ", medians_nonuni[-1])
    print("Median score for 17: ", medians_17[-1])
    print("Median score for 18-24: ", medians_24[-1])
    print("Median score for 25-44: ", medians_44[-1])
    print("Median score for 45-64: ", medians_64[-1])
    print("Median score for 65: ", medians_65[-1])
    print("Median score for men 18-24 uni: ", medians_men_24_uni[-1])
    print("Median score for men 25-44 uni: ", medians_men_44_uni[-1])
    print("Median score for women 45-64 nonuni: ", medians_women_64_nonuni[-1])

    # Compute the max deviation
    maximums.append(np.max(abs(medians_arab_gender[5:]-np.mean(medians_arab_gender))/np.mean(medians_arab_gender)))
    maximums.append(np.max(abs(medians_arab_uni[5:]-np.mean(medians_arab_uni))/np.mean(medians_arab_uni)))
    maximums.append(np.max(abs(medians_arab_age[5:]-np.mean(medians_arab_age))/np.mean(medians_arab_age)))
    maximums.append(np.max(abs(medians_men[5:]-np.mean(medians_men))/np.mean(medians_men)))
    maximums.append(np.max(abs(medians_women[5:]-np.mean(medians_women))/np.mean(medians_women)))
    maximums.append(np.max(abs(medians_uni[5:]-np.mean(medians_uni))/np.mean(medians_uni)))
    maximums.append(np.max(abs(medians_nonuni[5:]-np.mean(medians_nonuni))/np.mean(medians_nonuni)))
    maximums.append(np.max(abs(medians_17[5:]-np.mean(medians_17))/np.mean(medians_17)))
    maximums.append(np.max(abs(medians_24[5:]-np.mean(medians_24))/np.mean(medians_24)))
    maximums.append(np.max(abs(medians_44[5:]-np.mean(medians_44))/np.mean(medians_44)))
    maximums.append(np.max(abs(medians_64[5:]-np.mean(medians_64))/np.mean(medians_64)))
    maximums.append(np.max(abs(medians_65[5:]-np.mean(medians_65))/np.mean(medians_65)))
    # print("Max of Maxs", np.max(maximums))
    # print("Mean of max", np.mean(maximums))


    # Plotting the evolution of the medians of the assimilation scores
    #  Gender
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.plot(nb_ints, medians_arab_gender, 'b')
    rects2 = ax.plot(nb_ints, medians_men, 'g-*')
    rects3 = ax.plot(nb_ints, medians_women, 'r-o')
    ax.set_xlabel("Number of interests")
    ax.set_ylabel("Assimilation score")
    ax.set_ylim(y_low, y_high)
    ax.legend((rects3[0], rects2[0], rects1[0]),
              ("Target = " + arab_pops_c[2], "Target = " + arab_pops_c[1], "Target = " + arab_pops_c[0]),
              loc="lower right", prop={'size': 10})

    plt.savefig("final_gender_evolution.pdf", format='pdf', bbox_inches='tight')

    # Education
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.plot(nb_ints, medians_arab_uni, 'k')
    rects4 = ax.plot(nb_ints, medians_uni, 'g-*')
    rects5 = ax.plot(nb_ints, medians_nonuni, 'r-o')
    ax.set_xlabel("Number of interests")
    ax.set_ylabel("Assimilation score")
    ax.set_ylim(y_low, y_high)
    ax.legend((rects5[0], rects4[0], rects1[0]),
               ("Target = " + arab_pops_c[4], "Target = " + arab_pops_c[3], "Target = " + arab_pops_c[0]), loc="lower right", prop={'size': 10})
    plt.savefig("final_education_evolution.pdf", format='pdf', bbox_inches='tight')


    # Age
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.plot(nb_ints, medians_arab_age, 'k')
    rects2 = ax.plot(nb_ints, medians_17, 'g-*')
    rects3 = ax.plot(nb_ints, medians_24, 'r-o')
    rects4 = ax.plot(nb_ints, medians_44, 'm-+')
    rects5 = ax.plot(nb_ints, medians_64, 'b-|')
    rects6 = ax.plot(nb_ints, medians_65, 'c-.')
    ax.set_xlabel("Number of interests")
    ax.set_ylabel("Assimilation score")
    ax.set_ylim(y_low, y_high)
    ax.legend((rects6[0], rects5[0], rects4[0], rects3[0], rects2[0], rects1[0]),
               ("Target = " + arab_pops_c[9], "Target = " + arab_pops_c[8], "Target = " + arab_pops_c[7],
                "Target = " + arab_pops_c[6], "Target = " + arab_pops_c[5], "Target = " + arab_pops_c[0]), loc="lower right", prop={'size': 10})
    plt.savefig("final_age_evolution.pdf", format='pdf', bbox_inches='tight')


    # Plotting the median of the assimilation scores for the largest number of interests
    # Gender
    fig = plt.figure()
    ind = np.arange(14)  # the y locations for the groups
    ax = fig.add_subplot(111)
    rects21 = ax.barh(ind[0], np.median(arab_score_0_age), height, color='k')
    rects22 = ax.barh(ind[1], np.median(arab_score_1), height, color=(0.25, 0, 0.51))
    rects23 = ax.barh(ind[2], np.median(arab_score_2), height, color=(0.55, 0, 0.81))

    # Uni
    rects24 = ax.barh(ind[3], np.median(arab_score_3), height, color=(0.65, 0.17, 0.17))
    rects25 = ax.barh(ind[4], np.median(arab_score_4), height, color=(0.85, 0.37, 0.37))

    # Age
    rects26 = ax.barh(ind[5], np.median(arab_score_5), height, color=(1, 0.28, 0))
    rects27 = ax.barh(ind[6], np.median(arab_score_6), height, color=(1, 0.4, 0))
    rects28 = ax.barh(ind[7], np.median(arab_score_7), height, color=(1, 0.52, 0))
    rects29 = ax.barh(ind[8], np.median(arab_score_8), height, color=(1, 0.64, 0))
    rects210 = ax.barh(ind[9], np.median(arab_score_9), height, color=(1, 0.86, 0))

    # Mixes
    rects28 = ax.barh(ind[10], np.median(arab_score_10), height, color=(0, 0.7, 0))
    rects29 = ax.barh(ind[11], np.median(arab_score_11), height, color=(0.3, 0.7, 0.3))
    rects210 = ax.barh(ind[12], np.median(arab_score_12), height, color=(0.5, 0.7, 0.5))

    # Bundesliga
    rects211 = ax.barh(ind[13], np.median(arab_score_13), height, color=(0, 0, 0.7))

    ax.set_xlabel("Assimilation score")
    ax.set_yticks(ind)
    ax.set_yticklabels([arab_pops_c[0], arab_pops_c[1], arab_pops_c[2], arab_pops_c[3], arab_pops_c[4],
                        arab_pops_c[5], arab_pops_c[6], arab_pops_c[7], arab_pops_c[8], arab_pops_c[9],
                        "Men, Uni. G., 18-24", "Men, Uni. G., 25-44", "Women, Not Uni. G., 45-64"])
    ax.set_xlim(x_low, x_high)
    plt.savefig("final_arab_subgroups.pdf", format='pdf', bbox_inches='tight')


    ################################
    # Assimilation score per state #
    ################################

    nb_pops = 3
    sel = [0, 4, 5]
    sel_pops = [pops[sel[0]], pops[sel[1]], pops[sel[2]]]
    sel_pops_c = [pops_c[sel[0]], pops_c[sel[1]], pops_c[sel[2]]]
    sel_col = [col[sel[0]], col[sel[1]], col[sel[2]]]
    scores_1, target_pop_sizes_1 = state_specific_scores(
                                        "interests_audiences/Arab_sp_exp_DE/",
                                        "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE",
                                        "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
                                        score_type, nb_int)
    scores_2, target_pop_sizes_2 = state_specific_scores(
                                        "interests_audiences/French_exp_DE/",
                                        "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE",
                                        "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
                                        score_type, nb_int)
    scores_3, target_pop_sizes_3 = state_specific_scores(
                                        "interests_audiences/Austrian_exp_DE/",
                                        "interests_audiences/20180103-001107_global_3088_interests_nonexp_DE",
                                        "interests_audiences/20180103-001107_global_3088_interests_nonexp_ArL.csv",
                                        score_type, nb_int)

    N = 16
    ind = np.arange(N)  # the y locations for the groups
    height = 0.3      # the width of the bars
    fig = plt.figure()
    median_scores_1 = []
    for i in scores_1:
        median_scores_1.append(np.median(i))
    median_scores_2 = []
    for i in scores_2:
        median_scores_2.append(np.median(i))
    median_scores_3 = []
    for i in scores_3:
        median_scores_3.append(np.median(i))
    print("Std of median " + sel_pops[0] + ": ", np.std(median_scores_1))
    print("Mean of median " + sel_pops[0] + ": ", np.mean(median_scores_1))
    print("Std of median " + sel_pops[1] + ": ", np.std(median_scores_2))
    print("Mean of median " + sel_pops[1] + ": ", np.mean(median_scores_2))
    print("Std of median " + sel_pops[2] + ": ", np.std(median_scores_3))
    print("Mean of median " + sel_pops[2] + ": ", np.mean(median_scores_3))
    print("")

    ax = fig.add_subplot(111)
    rects21 = ax.barh(ind, median_scores_1, height, color=sel_col[0])
    rects22 = ax.barh(ind+height, median_scores_2, height, color=sel_col[1])
    rects23 = ax.barh(ind+2*height, median_scores_3, height, color=sel_col[2])
    ax.set_xlabel('Median')
    ax.set_yticklabels(states_name)
    ax.legend((rects21[0], rects22[0], rects23[0]),
              ("Target = " + sel_pops_c[0], "Target = " + sel_pops_c[1],
               "Target = " + sel_pops_c[2]))


    ########################################################################
    # Correlation between the number of refugees and Arab speaking ex-pats #
    ########################################################################

    fig = plt.figure()
    state_area_km = [35752, 70552, 892, 29479, 419, 755, 21115, 47609,
                     23180, 34085, 19853, 2569, 18416, 20446, 15799,
                     16172]
    target_per_km = np.asarray(target_pop_sizes_1)/np.asarray(state_area_km)
    refugees_per_km = np.asarray([3.96, 2.37, 61.84, 1.13, 24.47, 36.57, 3.79, 2.15,
                       0.96, 6.80, 2.66, 1.53, 3.02, 2.34, 5.19, 1.85])

    print("Correlation coefficient , ", np.corrcoef([refugees_per_km, target_per_km])[0][1])

    ax = fig.add_subplot(111)
    rect1 = ax.plot(range(16), refugees_per_km)
    rect2 = ax.plot(range(16), target_per_km)
    plt.ylabel("Number of migrants\n per square kilometers")
    plt.xlabel("German states")
    ax.legend((rect1[0], rect2[0]),
              ("Estimate from the Brookings Institution",
               "Estimate using the Facebook API with \ntarget being Arabic speaking expats"), loc="upper right", prop={'size': 10})
    plt.savefig("arab_migrants_per_km_comp.pdf", format='pdf', bbox_inches='tight')


    #####################################
    # Correlation with election results #
    #####################################

    afd_results = np.asarray([12.2, 12.4, 12.0, 20.2, 10.0, 7.8, 11.9, 18.6, 9.1,
                              9.4, 11.2, 10.1, 27.0, 19.6, 8.2, 22.7])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(afd_results, median_scores_1)
    ax.set_xlabel("Afd results per state (%)")
    ax.set_ylabel("median(CIR for Locals " + score_type + " CIR for Arab speaking expats)")
    ax.set_title("Median")

    print("Arab Median correlation coefficient , ", np.corrcoef(afd_results, median_scores_1)[0][1])

    # plt.show()
