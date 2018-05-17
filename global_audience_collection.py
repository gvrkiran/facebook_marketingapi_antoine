import json
import main
import logging
import time
import sys


# Simplify the writing of the interest
def extract_name(json_string):
    name = "Population size"
    if json_string:
        name = json_string['name'][0]
    return name


def extract_code(json_string):
    code = 'None'
    if json_string:
        # code = json_string['and'][0]
        code = json_string['or'][0]
    return code


time_stamp = time.strftime("%Y%m%d-%H%M%S")

# Fetching parameters of the script
if len(sys.argv) != 9:
    print("ERROR: You must enter 8 values: nb_interests, interest_file_num, country, bhv, lang,"
          " age, gen, schol")
    print("       You entered {}".format(len(sys.argv)-1))
    sys.exit(-1)

nb_interests = int(sys.argv[1])
interest_file_num = int(sys.argv[2])
loc = int(sys.argv[3])
bhv = int(sys.argv[4])
lang = int(sys.argv[5])
age = int(sys.argv[6])
gen = int(sys.argv[7])
schol = int(sys.argv[8])

watcherAPI = main.PySocialWatcher
logging.basicConfig(level=logging.INFO)
watcherAPI.load_credentials_file("credentials.csv")
# watcherAPI.check_tokens_account_valid()

target_loc_dict = {0: {"name": "countries", "values": ["DE"]},
                   1: {"name": "countries", "values": ["KM", "DZ", "BH", "IQ", "DJ", "EG", "JO", "LB", "LY", "KW",
                                                       "MA", "QA", "OM", "SA", "PS", "MR", "SO", "TN", "SS", "YE",
                                                       "AE"]},
                   2: {"name": "countries", "values": ["FR"]},
                   3: {"name": "countries", "values": ["AT"]},
                   4: {"name": "countries", "values": ["ES"]},
                   5: {"name": "countries", "values": ["TR"]}}

target_loc_str_dict = {0: "_DE",
                       1: "_ArL",
                       2: "_FR",
                       3: "_AT",
                       4: "_ES",
                       5: "_TR"}

target_loc = target_loc_dict[loc]
target_loc_str = target_loc_str_dict[loc]


# Arab expats or not or from somewhere else
target_bhv_dict = {0: None,
                   1: {"or": [6015559470583], "name": "Ex-pats (All)"},
                   2: {"not": [6015559470583], "name": "Not Expats"},
                   3: {"or": [6019367014383], "name": "Ex-pats (France)"},
                   4: {"or": [6023675997383], "name": "Ex-pats (Austria)"},
                   5: {"or": [6023287393783], "name": "Ex-pats (the Netherlands)"},
                   6: {"or": [6023620475783], "name": "Ex-pats (United States)"},
                   7: {"or": [6019396654583], "name": "Ex-pats (Italy)"},
                   8: {"or": [6019366943583], "name": "Ex-pats (Spain)"},
                   9: {"or": [6021354152983], "name": "Expats (United Kingdom)"}}

target_bhv_str_dict = {0: "",
                       1: "_exp",
                       2: "_nonexp",
                       3: "_Frech_exp",
                       4: "_Austrian_exp",
                       5: "_Ndl_exp",
                       6: "_American_exp",
                       7: "_Italian_exp",
                       8: "_Spanish_exp",
                       9: "_English_exp"}


target_bhv = target_bhv_dict[bhv]
target_bhv_str = target_bhv_str_dict[bhv]

# Language
target_lang_dict = {0: None,
                   1: {"values": [28], "name": "Arabic"},
                   2: {"values": [19], "name": "Turkish"}}

target_lang_str_dict = {0: "",
                        1: "_Arab_sp",
                        2: "_Turkish_sp"}

target_lang = target_lang_dict[lang]
target_lang_str = target_lang_str_dict[lang]

# Education level
target_schol_dict = {0: None,
                     1: {"name": "University graduate", "or": [3]},
                     2: {"name": "Not university graduate", "not": [3]}}

target_schol_str_dict = {0: "",
                         1: "_uni",
                         2: "_nonuni"}

target_schol = target_schol_dict[schol]
target_schol_str = target_schol_str_dict[schol]

# Gender
target_gen_str_dict = {0: "",
                       1: "_men",
                       2: "_women"}

target_gen = gen
target_gen_str = target_gen_str_dict[gen]

# Age
target_age_dict = {0: {"min": 18, "max": 65},
                   1: {"max": 17},
                   2: {"min": 18, "max": 24},
                   3: {"min": 25, "max": 44},
                   4: {"min": 45, "max": 64},
                   5: {"min": 65}}

target_age_str_dict = {0: "",
                       1: "_17",
                       2: "_18_24",
                       3: "_25_44",
                       4: "_45_64",
                       5: "_65"}

target_age = target_age_dict[age]
target_age_str = target_age_str_dict[age]

'''
Creating the request
'''
# The commented lines show an example of how to get a population with a certain interest
# interest_list = [{"and": [6003354236021], "name": ["Bundesliga"]}]
interest_list = [None]
if interest_file_num == 0:
    interest_file = "top_interests_2.json"
elif interest_file_num == 1:
    interest_file = "top_interests.json"
elif interest_file_num == 2:
    interest_file = "arab_interests.json"
else:
    interest_file = "german_interests.json"

with open(interest_file) as data_file:
    data = json.load(data_file)
interests = data['data']
count = 1
for x in interests:
    if count <= nb_interests:
        # interest_list.append({"and": [x['id'], 6003354236021], "name": [x['name'] + " and Bundesliga"]})
        interest_list.append({"or": [x['id']], "name": [x['name']]})
    else:
        break
    count += 1

dic_target = {
    "name": "Audiences for " + target_lang_str + target_bhv_str + target_age_str + target_gen_str + target_schol_str
                             + target_loc_str,
    "publisher_platforms": ["facebook"],
    "geo_locations": [target_loc],
    "languages": [target_lang],
    "ages_ranges": [target_age],
    "genders": [target_gen],
    "interests": interest_list,
    "behavior": [target_bhv],
    "scholarities": [target_schol]
}

'''
Collecting interests
'''
json.dump(dic_target, open("./input_examples/" + time_stamp + "_global_audience_collection.json", "w"),
          indent=2, separators=(',', ': '))
data_target = watcherAPI.run_data_collection("./input_examples/" + time_stamp + "_global_audience_collection.json")

'''
Process the data to keep only what is needed (name, code and audience)
'''
# Get a data frame with interest name and frequencies for Germany and Arab League countries
interests = data_target[['interests', 'audience']]

# Simplify writing of interests
interests['name'] = interests['interests'].apply(extract_name)
interests['code'] = interests['interests'].apply(extract_code)
interests = interests[['name', 'code', 'audience']]

# Save the file
title = "./results/" + time_stamp + "_global_" + str(interests['name'].shape[0]-1) + "_interests" \
                     + target_lang_str + target_bhv_str + target_age_str + target_gen_str + target_schol_str \
                     + target_loc_str  # + "_bundesliga"

interests.to_csv(title + ".csv")
