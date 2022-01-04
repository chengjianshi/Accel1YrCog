**LONG TO WIDE FORMAT**
 
 
// Drop study participants not enrolled in FACE Aging study
drop if facestudy==0

// Recreate survey_type variable to add "_#" to end of new vars
gen tmp="_"
gen time = .
	replace time = 00 if redcap_event_name == "baseline_arm_1"
	replace time = 06 if redcap_event_name == "6_month_followup_arm_1"
	replace time = 12 if redcap_event_name == "12_month_followup_arm_1"
	replace time = 18 if redcap_event_name == "18_month_followup_arm_1"
	replace time = 24 if redcap_event_name == "24_month_followup_arm_1"
	replace time = 30 if redcap_event_name == "30_month_followup_arm_1"
	replace time = 36 if redcap_event_name == "36_month_followup_arm_1"
	replace time = 42 if redcap_event_name == "42_month_followup_arm_1"
	replace time = 48 if redcap_event_name == "48_month_followup_arm_1"
	replace time = 60 if redcap_event_name == "60_month_followup_arm_1"

// Keep only primary annual clinical evaluations, drop short 6-month check in visits
drop if redcap_event_name == "6_month_followup_arm_1"
drop if redcap_event_name == "18_month_followup_arm_1"
drop if redcap_event_name == "30_month_followup_arm_1"
drop if redcap_event_name == "42_month_followup_arm_1"
drop if redcap_event_name == "48_month_followup_arm_1"
drop if redcap_event_name == "60_month_followup_arm_1"

// Keep analytic variables
keep participant_number survey_date survey_date_and_deta_v_0 heart_attack ///
congestive periph_vasc_dis cereb_vasc_dis cereb_vasc_dis_dif chron_pulm_dis chron_pulm_dis_med ///
pep_ulc_dis liver_dis liver_dis_spec diabetes diabetes_spec renal_dis con_tis_dis alzheimers leukemia lymphoma ///
cancer cancer_metastatic aids charlson_score hypothyroid osa baseline_hgb baseline_cr ///
gfr_within_3_months_of_int baselinemeds two_falls_in_12_mo fall1 fall2 ///
balance_difficulty positive_fall_screen er_visits_12_mo hospital_admit_12_mo hospitalization ///
hospitalization2 nh_12_mo nh_or_rehab___1 nh_or_rehab___2 nh_or_rehab___3 nh_or_rehab___4 nh_enter nh_leave ///
palliative ps_score six_item_minnesota_l_v_4 ///
yes_caregiver caregivers___1 caregivers___2 caregivers___3 caregivers___4 caregivers___5 caregivers___6 caregivers___7 caregivers___8 homemaker birth_date age hispanic race gender education housing ///
home_dwellers___1 home_dwellers___2 home_dwellers___3 home_dwellers___4 home_dwellers___5 total_adult_persons total_children total_number_in_household relationship ///
monthly_individ_income other_hh_financial_support monthly_house_income smoke alcohol height weight bmi sbp dbp hearing_impairments visual_impairment ///
incontinence depends cane have_cane cane_use walker have_walker walker_use wheelchair have_wheelchair ///
wheelchair_use scooter have_scooter scooter_use depression_1 depression_2 dep_calc ///
how_many_hours_of_sleep_do ///
pain_man_av trails_score cube_score clock_contour clock_numbers ///
clock_hands naming_lion naming_rhino naming_camel repeated_digits_forward repeated_digits_backward ///
finger_tap_score serial_sevens_score sentence_john sentence_cats_dogs fluency_score train_bicycle ///
watch_ruler face velvet church daisy red date month year day place city weight_year_ago weight_now ///
exhaust_effort exhaust_going grip1 grip2 grip3 ///
walk_score height_cm wlk_1_attmptd walk8_timesec ///
phys_activitykcals phys_score total_frailtyscore sbys_stand sem_tand_stand tand_stand c_stnd_5_time_attmpt chair_stnd_sec hospice_date date_of_death /// 
too_much_time tires_me too_far embarrassed cost schedules fatigue spouse relationships clothes encourage responsibilities hard_work too_few_places physically better_mood exercise muscles activity_i_enjoy accomplishment mentally endurance strengthen_my_bones good routine enjoy community quit stretching hurt ///
faster_walk_points phys_perf_score dropout dropout_reason dropout_date ///
walk_pleasure walk_pleasure_times walk_pleasure_min stairs stairs_times stairs_min bike bike_times bike_min home_exercise home_exercise_times home_exercise_min health_club health_club_times health_club_min golf golf_times golf_min mowed_lawn mowed_lawn_times mowed_lawn_min weed_garden weed_garden_times weed_garden_min spade_garden spade_garden_times spade_garden_min rake_lawn rake_lawn_times rake_lawn_min shovel_snow shovel_snow_times shovel_snow_min swimming swimming_time swimming_min walk_work walk_work_times walk_work_min xc_hiking xc_hiking_times xc_hiking_min dance dance_times dance_min weights weights_times weights_mi carpentry carpentry_time carpentry_min paint paint_times paint_min other_activity_1 other_activity1_times other_activity1_min walk_month walk_minutes walk_num_sessions garden_month garden_minutes garden_num_session gen_ex_month gen_ex_minutes gen_ex_num_session hh_chores_month hh_chores_num_month hh_chores_num_sessions hh_chores_minutes walk_num_month garden_num_month gen_ex_num_month mow_month mow_num_month mow_num_session mow_minutes golf_month golf_num_month golf_num_session golf_minutes ///
health stooping lifting reaching writing walking heavyhouse shop manage_money walkroom lighthouse bathe ves_age ///
AGE povertyind black white otherrace highschl posdep depressed BMI heightm BMIr bmicat5 underwt normwt overwt ///
obese1 obese2 obese3 obese bmisq wtlosslb wtlosspercent poswtloss exhausteffort exhaustgoing ///
posexhaust avegrip posgrip avewalk15 avewalk15r posgait totalwalkMET2 totalgardenMET2 totalgenexerMET2 ///
totalchoresMET2 totalmowMET2 totalMET2 posphyact2 frailty nonfrail prefrail frail preorfrail balancescore ///
invbalancescore balancetime invbalancetime badbalanceobj balancediscordsubjbad balancediscord posAGSfallsc ///
poorhealth stoopingdiff liftingdiff reachingdiff writingdiff walkingdiff heavyhousediff vesscore1 vesscore1b ///
shopdiff manage_moneydiff walkroomdiff lighthousediff bathediff vesscore2 vesagescore vestotalscore fairpoor ///
moca_score_cal chron_pulm_dis_tx liver_dis_sev diabetes_sev renal_dis_sev leukemia_sev lymphoma_sev ///
cancer_sev cereb_vasc_dis_sev charlson_cal multimorbid cogimp cogimp_educ sbys_stand2 sem_tand_stand2 ///
tand_stand2 obeseage obeseves obesepain obesecharl obesemoca /// 
time beta_blocker non_beta_hyperten anti_cholinergic opioids anti_psych anti_depress benzo cholesterol_low nsaids ca_block_non_dihydro ca_block_dihydro anti_coag prednisone inccat baselinemeds_inc charlson_cal_inc medcat

// Reshape to wide dataset
reshape wide survey_date survey_date_and_deta_v_0 heart_attack ///
congestive periph_vasc_dis cereb_vasc_dis cereb_vasc_dis_dif chron_pulm_dis chron_pulm_dis_med ///
pep_ulc_dis liver_dis liver_dis_spec diabetes diabetes_spec renal_dis con_tis_dis alzheimers leukemia lymphoma ///
cancer cancer_metastatic aids charlson_score hypothyroid osa baseline_hgb baseline_cr ///
gfr_within_3_months_of_int baselinemeds two_falls_in_12_mo fall1 fall2 ///
balance_difficulty positive_fall_screen er_visits_12_mo hospital_admit_12_mo hospitalization ///
hospitalization2 nh_12_mo nh_or_rehab___1 nh_or_rehab___2 nh_or_rehab___3 nh_or_rehab___4 nh_enter nh_leave ///
palliative ps_score six_item_minnesota_l_v_4 ///
yes_caregiver caregivers___1 caregivers___2 caregivers___3 caregivers___4 caregivers___5 caregivers___6 caregivers___7 caregivers___8 homemaker birth_date age hispanic race gender education housing ///
home_dwellers___1 home_dwellers___2 home_dwellers___3 home_dwellers___4 home_dwellers___5 total_adult_persons total_children total_number_in_household relationship ///
monthly_individ_income other_hh_financial_support monthly_house_income smoke alcohol height weight bmi sbp dbp hearing_impairments visual_impairment ///
incontinence depends cane have_cane cane_use walker have_walker walker_use wheelchair have_wheelchair ///
wheelchair_use scooter have_scooter scooter_use depression_1 depression_2 dep_calc ///
how_many_hours_of_sleep_do ///
pain_man_av trails_score cube_score clock_contour clock_numbers ///
clock_hands naming_lion naming_rhino naming_camel repeated_digits_forward repeated_digits_backward ///
finger_tap_score serial_sevens_score sentence_john sentence_cats_dogs fluency_score train_bicycle ///
watch_ruler face velvet church daisy red date month year day place city weight_year_ago weight_now ///
exhaust_effort exhaust_going grip1 grip2 grip3 ///
walk_score height_cm wlk_1_attmptd walk8_timesec ///
phys_activitykcals phys_score total_frailtyscore sbys_stand sem_tand_stand tand_stand c_stnd_5_time_attmpt chair_stnd_sec hospice_date date_of_death /// 
too_much_time tires_me too_far embarrassed cost schedules fatigue spouse relationships clothes encourage responsibilities hard_work too_few_places physically better_mood exercise muscles activity_i_enjoy accomplishment mentally endurance strengthen_my_bones good routine enjoy community quit stretching hurt ///
faster_walk_points phys_perf_score dropout dropout_reason dropout_date ///
walk_pleasure walk_pleasure_times walk_pleasure_min stairs stairs_times stairs_min bike bike_times bike_min home_exercise home_exercise_times home_exercise_min health_club health_club_times health_club_min golf golf_times golf_min mowed_lawn mowed_lawn_times mowed_lawn_min weed_garden weed_garden_times weed_garden_min spade_garden spade_garden_times spade_garden_min rake_lawn rake_lawn_times rake_lawn_min shovel_snow shovel_snow_times shovel_snow_min swimming swimming_time swimming_min walk_work walk_work_times walk_work_min xc_hiking xc_hiking_times xc_hiking_min dance dance_times dance_min weights weights_times weights_mi carpentry carpentry_time carpentry_min paint paint_times paint_min other_activity_1 other_activity1_times other_activity1_min walk_month walk_minutes walk_num_sessions garden_month garden_minutes garden_num_session gen_ex_month gen_ex_minutes gen_ex_num_session hh_chores_month hh_chores_num_month hh_chores_num_sessions hh_chores_minutes walk_num_month garden_num_month gen_ex_num_month mow_month mow_num_month mow_num_session mow_minutes golf_month golf_num_month golf_num_session golf_minutes ///
health stooping lifting reaching writing walking heavyhouse shop manage_money walkroom lighthouse bathe ves_age ///
AGE povertyind black white otherrace highschl posdep depressed BMI heightm BMIr bmicat5 underwt normwt overwt ///
obese1 obese2 obese3 obese bmisq wtlosslb wtlosspercent poswtloss exhausteffort exhaustgoing ///
posexhaust avegrip posgrip avewalk15 avewalk15r posgait totalwalkMET2 totalgardenMET2 totalgenexerMET2 ///
totalchoresMET2 totalmowMET2 totalMET2 posphyact2 frailty nonfrail prefrail frail preorfrail balancescore ///
invbalancescore balancetime invbalancetime badbalanceobj balancediscordsubjbad balancediscord posAGSfallsc ///
poorhealth stoopingdiff liftingdiff reachingdiff writingdiff walkingdiff heavyhousediff vesscore1 vesscore1b ///
shopdiff manage_moneydiff walkroomdiff lighthousediff bathediff vesscore2 vesagescore vestotalscore fairpoor ///
moca_score_cal chron_pulm_dis_tx liver_dis_sev diabetes_sev renal_dis_sev leukemia_sev lymphoma_sev ///
cancer_sev cereb_vasc_dis_sev charlson_cal multimorbid cogimp cogimp_educ sbys_stand2 sem_tand_stand2 ///
tand_stand2 obeseage obeseves obesepain obesecharl obesemoca /// 
beta_blocker non_beta_hyperten anti_cholinergic opioids anti_psych anti_depress benzo cholesterol_low nsaids ca_block_non_dihydro ca_block_dihydro anti_coag prednisone inccat baselinemeds_inc charlson_cal_inc medcat, i(participant_number) j(time)

