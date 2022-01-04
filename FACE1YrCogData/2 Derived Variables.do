*NEW VARIABLES

label define exercise_you_prefer___1__ 0 "No" 1 "Any exercise using a machine (treadmill, elliptical, stationary bicycle, etc)" 
label define exercise_you_prefer___2__ 0 "No" 1 "Walking" 
label define exercise_you_prefer___3__ 0 "No" 1 "Gardening" 
label define exercise_you_prefer___4__ 0 "No" 1 "Golfing" 
label define exercise_you_prefer___5__ 0 "No" 1 "Stretching" 
label define exercise_you_prefer___6__ 0 "No" 1 "Yoga/Tai Chi" 
label define exercise_you_prefer___7__ 0 "No" 1 "Weight lifting or resistance exercises" 
label define exercise_you_prefer___8__ 0 "No" 1 "Exercises I can do in my home" 
label define exercise_you_prefer___9__ 0 "No" 1 "Exercises I can do from my chair" 
label define exercise_you_prefer___10__ 0 "No" 1 "Cleaning my house" 
label define exercise_you_prefer___11__ 0 "No" 1 "Mowing the lawn or shoveling snow" 
label define exercise_you_prefer___12__ 0 "No" 1 "Swimming or water exercises" 
label define exercise_you_prefer___13__ 0 "No" 1 "Aerobics" 
label define exercise_you_prefer___14__ 0 "No" 1 "Exercises or activity in a group" 
label define exercise_you_prefer___15__ 0 "No" 1 "Exercises or activity with one other person" 
label define exercise_you_prefer___16__ 0 "No" 1 "Exercises or activity alone" 
label define exercise_you_prefer___17__ 0 "No" 1 "Exercises or activity led by an instructor on a video or DVD" 
label define exercise_you_prefer___18__ 0 "No" 1 "Checked" 
label define activity_resources___1__ 0 "No" 1 "I have friends or family members caregivers that would help me be active" 
label define activity_resources___2__ 0 "No" 1 "I have exercise equipment (bicycle, treadmill, weights, elliptical, therapy bands, exercise videos / DVD's, aerobic step, etc) at home" 
label define activity_resources___3__ 0 "No" 1 "I can get to a gym" 
label define activity_resources___4__ 0 "No" 1 "I can get to an activity class geared for adults 65 years old and older" 
label define activity_resources___5__ 0 "No" 1 "I can get to a walking path" 
label define activity_resources___6__ 0 "No" 1 "I am able to follow instructions for exercise / activity" 
label define activity_resources___7__ 0 "No" 1 "I enjoy being active." 
label define activity_resources___8__ 0 "No" 1 "Other" 

label values exercise_you_prefer___1 exercise_you_prefer___1__
label values exercise_you_prefer___2 exercise_you_prefer___2__
label values exercise_you_prefer___3 exercise_you_prefer___3__
label values exercise_you_prefer___4 exercise_you_prefer___4__
label values exercise_you_prefer___5 exercise_you_prefer___5__
label values exercise_you_prefer___6 exercise_you_prefer___6__
label values exercise_you_prefer___7 exercise_you_prefer___7__
label values exercise_you_prefer___8 exercise_you_prefer___8__
label values exercise_you_prefer___9 exercise_you_prefer___9__
label values exercise_you_prefer___10 exercise_you_prefer___10__
label values exercise_you_prefer___11 exercise_you_prefer___11__
label values exercise_you_prefer___12 exercise_you_prefer___12__
label values exercise_you_prefer___13 exercise_you_prefer___13__
label values exercise_you_prefer___14 exercise_you_prefer___14__
label values exercise_you_prefer___15 exercise_you_prefer___15__
label values exercise_you_prefer___16 exercise_you_prefer___16__
label values exercise_you_prefer___17 exercise_you_prefer___17__
label values exercise_you_prefer___18 exercise_you_prefer___18__
label values activity_resources___1 activity_resources___1__
label values activity_resources___2 activity_resources___2__
label values activity_resources___3 activity_resources___3__
label values activity_resources___4 activity_resources___4__
label values activity_resources___5 activity_resources___5__
label values activity_resources___6 activity_resources___6__
label values activity_resources___7 activity_resources___7__
label values activity_resources___8 activity_resources___8__

gen survey_date_new = date(survey_date, "MDY")
	format survey_date_new %td
	drop survey_date
	rename survey_date_new survey_date
gen birth_date_new = date(birth_date, "MDY")
	format birth_date_new %td
	drop birth_date
	rename birth_date_new birth_date
gen _date_ = date(fall1,"MDY")
	drop fall1
	rename _date_ fall1
	format fall1 %td
gen _date_ = date(fall2,"MDY")
	drop fall2
	rename _date_ fall2
	format fall2 %td
gen _date_ = date(hospitalization,"MDY")
	drop hospitalization
	rename _date_ hospitalization
	format hospitalization %td
gen _date_ = date(hospitalization2,"MDY")
	drop hospitalization2
	rename _date_ hospitalization2
	format hospitalization2 %td
gen _date_ = date(nh_enter,"MDY")
	drop nh_enter
	rename _date_ nh_enter
	format nh_enter %td
gen _date_ = date(nh_leave,"MDY")
	drop nh_leave
	rename _date_ nh_leave
	format nh_leave %td
gen _date_ = date(dropout_date,"MDY")
	drop dropout_date
	rename _date_ dropout_date
	format dropout_date %td
gen _date_ = date(hospice_date,"MDY")
	drop hospice_date
	rename _date_ hospice_date
	format hospice_date %td
gen _date_ = date(date_of_death,"MDY")
	drop date_of_death
	rename _date_ date_of_death
	format date_of_death %td

//For FACE Aging study participants only:
gen facestudy = 0
	replace facestudy = 1 if participant_number >= 7 & participant_number <=9 | participant_number==22 | participant_number==116 | participant_number >=141
	
//Demographics
*Age
gen AGE = .
replace AGE = round((survey_date - birth_date)/365.25,0.1)

gen povertyind = .
	replace povertyind = 1 if monthly_individ_income <= 907.5
	replace povertyind = 0 if monthly_individ_income >907.5 & monthly_individ_income < .
	replace povertyind = 2 if monthly_individ_income==.
egen inccat = cut(monthly_individ_income), at(0, 2000, 4000, 6000, 99999)
	replace inccat=100000 if monthly_individ_income==.
label def inccat 0 "0<$2000/month" 2000 "$2000-3999/month" 4000 "$4000-5999/month" 6000 "$6000+" 100000 "Missing"
label val inccat inccat

*Race	
gen black = .
	replace black = 1 if race == 2
	replace blac = 0 if race < 2 | race > 2 & race != .
gen white = .
	replace white = 1 if race == 1
	replace white = 0 if race > 1 & race != .
recode race min/1=0 2/2=1 3/15=2 16/max=., gen(racecat)
gen race_r = race
	replace race_r = . if race >= 16
/* Other race includes anything other than black/white here*/
gen otherrace = .
	replace otherrace = 1 if race > 2 & race != .
	replace otherrace = 0 if race <= 2
	
*Education
gen highschl = .
	replace highschl = 1 if education >= 4 & education !=.
	replace highschl = 0 if education < 4
gen college = .
	replace college = 1 if education >=5 & education !=.
	replace college = 0 if education < 5

*Depression
gen posdep = .
	replace posdep = 1 if dep_calc >= 3 & dep_calc < .
	replace posdep = 0 if dep_calc < 3
gen depressed = .
	replace depressed = 1 if dep_calc >=3 & dep_calc!=.
	replace depressed = 0 if dep_calc <3 

*Cognition
gen cogimp = 1 if moca_score < 26
	replace cogimp = 0 if moca_score >=26 & moca_score <.
gen cogimp_educ = 1 if moca_score < 26
	replace cogimp_educ = 0 if moca_score >=26 & moca_score <.

*Body Mass Index
gen heightm = .
	replace heightm = height_cm/100
gen BMI = .
	replace BMI = (weight/(heightm*heightm))
gen BMIr = .
	replace BMIr = round(BMI, 0.1)
recode bmi min/17.999999=0 17.9999991/24.999999=1 24.9999991/29.999999=2 29.9999991/34.999999=3 34.9999991/39.999999=4 39.9999991/max=5, gen(bmicat5)
gen underwt = .
	replace underwt = 1 if bmicat5 == 0
	replace underwt = 0 if bmicat5 >= 1 & bmicat5 <.
gen normwt = .
	replace normwt = 1 if bmicat5 == 1
	replace normwt = 0 if bmicat5 == 0
	replace normwt = 0 if bmicat5 > 1 & bmicat5 <.
gen overwt = .
	replace overwt = 1 if bmicat5 == 2
	replace overwt = 0 if bmicat5 == 0
	replace overwt = 0 if bmicat5 == 1
	replace overwt = 0 if bmicat5 > 2 & bmicat5 <.
gen obese1 = .
	replace obese1 = 1 if bmicat5 == 3
	replace obese1 = 0 if bmicat5 == 0
	replace obese1 = 0 if bmicat5 == 1
	replace obese1 = 0 if bmicat5 == 2
	replace obese1 = 0 if bmicat5 > 3 & bmicat5 <.
gen obese2 = .
	replace obese2 = 1 if bmicat5 == 4
	replace obese2 = 0 if bmicat5 == 0
	replace obese2 = 0 if bmicat5 == 1
	replace obese2 = 0 if bmicat5 == 2
	replace obese2 = 0 if bmicat5 == 3
	replace obese2 = 0 if bmicat5 > 4 & bmicat5 <.
gen obese3 = .
	replace obese3 = 1 if bmicat5 == 5
	replace obese3 = 0 if bmicat5 < 5
gen obese = .
	replace obese = 1 if BMI >=30 & BMI !=.
	replace obese = 0 if BMI <30
gen bmisq = BMI*BMI

*Frailty
gen wtlosslb = .
	replace wtlosslb = round((weight_now - weight_year_ago), .1)
gen wtlosspercent = .
	replace wtlosspercent = round(((wtlosslb/weight_year_ago)*100), .1)
gen poswtloss = .
	replace poswtloss = 0 if wtlosspercent > -5.0 & wtlosspercent < 500
	replace poswtloss = 0 if wtlosslb >= -10.0 & wtlosslb < 500
	replace poswtloss = 1 if wtlosspercent <= -5.0
	replace poswtloss = 1 if wtlosslb <= -10.0
	recode poswtloss . = 0
gen exhausteffort = .
	replace exhausteffort = 1 if exhaust_effort >= 2 & exhaust_effort < .
	replace exhausteffort = 0 if exhaust_effort <2
gen exhaustgoing = .
	replace exhaustgoing = 1 if exhaust_going >= 2 & exhaust_going < .
	replace exhaustgoing = 0 if exhaust_going <2
gen posexhaust = .
	replace posexhaust = 1 if exhaustgoing == 1 | exhausteffort == 1
	replace posexhaust = 0 if exhaustgoing == 0 & exhausteffort == 0
	recode posexhaust . = 0
gen avegrip = .
	replace avegrip = (grip1 + grip2 + grip3)/3 if grip1 != . & grip2 != . & grip3 != .
	replace avegrip = (grip1 + grip2)/2 if grip1 !=. & grip2 !=. & grip3 ==.
	replace avegrip = grip1 if grip1 !=. & grip2 == . & grip3 == .
gen posgrip = .
	replace posgrip = 1 if gender == 0 & BMI <=24 & avegrip <=29
	replace posgrip = 0 if gender == 0 & BMI <=24 & avegrip >29
	replace posgrip = 1 if gender == 0 & BMI >24 & BMI <=26 & avegrip <=30
	replace posgrip = 0 if gender == 0 & BMI >24 & BMI <=26 & avegrip >30
	replace posgrip = 1 if gender == 0 & BMI >26 & BMI <=28 & avegrip <=31
	replace posgrip = 0 if gender == 0 & BMI >26 & BMI <=28 & avegrip >31
	replace posgrip = 1 if gender == 0 & BMI >28 & BMI !=. & avegrip <=32
	replace posgrip = 0 if gender == 0 & BMI >28 & BMI !=. & avegrip >32
	replace posgrip = 1 if gender == 1 & BMI <=23 & avegrip <=17
	replace posgrip = 0 if gender == 1 & BMI <=23 & avegrip >17
	replace posgrip = 1 if gender == 1 & BMI >23 & BMI <=26 & avegrip <=17.3
	replace posgrip = 0 if gender == 1 & BMI >23 & BMI <=26 & avegrip >17.3
	replace posgrip = 1 if gender == 1 & BMI >26 & BMI <=29 & avegrip <=18
	replace posgrip = 0 if gender == 1 & BMI >26 & BMI <=29 & avegrip >18
	replace posgrip = 1 if gender == 1 & BMI >29 & BMI !=. & avegrip <=21
	replace posgrip = 0 if gender == 1 & BMI >29 & BMI !=. & avegrip >21
recode posgrip . = 0
gen avewalk15 = .
	replace avewalk15 = (walk15_timesec + walk15_timesec2)/2 if walk15_timesec != . & walk15_timesec2 != .
	replace avewalk15 = walk15_timesec if walk15_timesec !=. & walk15_timesec2 ==. 
	replace avewalk15 = walk15_timesec2 if walk15_timesec2 !=. & walk15_timesec ==.
gen avewalk15r = .
	replace avewalk15r = round(avewalk15, 0.1)
gen posgait = .
	replace posgait = 1 if gender == 0 & height <= 68.1 & avewalk15r >=7 
	replace posgait = 0 if gender == 0 & height <= 68.1 & avewalk15r <7
	replace posgait = 1 if gender == 0 & height > 68.1 & avewalk15r >=6
	replace posgait = 0 if gender == 0 & height > 68.1 & avewalk15r <6
	replace posgait = 1 if gender == 1 & height <= 62.6 & avewalk15r >=7 
	replace posgait = 0 if gender == 1 & height <= 62.6 & avewalk15r <7
	replace posgait = 1 if gender == 1 & height > 62.6 & avewalk15r >=6
	replace posgait = 0 if gender == 1 & height > 62.6 & avewalk15r <6
recode posgait . = 0
/*Calculating kcal/week from Eckel's 6-Item Minnesota Leisure Time PAQ*/
gen totalwalkMET2 = 0
	replace totalwalkMET2 = (walk_month*3.5*weight)*(walk_minutes/60)*(walk_num_sessions/4)* (walk_num_month/12)
recode totalwalkMET2 . = 0
gen totalgardenMET2 = 0
	replace totalgardenMET2 = (garden_month*3.5*weight)*(garden_minutes/60)*(garden_num_session/4)* (garden_num_month/12)
recode totalgardenMET2 . = 0
gen totalgenexerMET2 = 0
	replace totalgenexerMET2 = (gen_ex_month*3.5*weight)*(gen_ex_minutes/60)*(gen_ex_num_session/4)* (gen_ex_num_month/12)
recode totalgenexerMET2 . = 0
gen totalchoresMET2 = 0
	replace totalchoresMET2  = (hh_chores_month*3.5*weight)*(hh_chores_minutes/60)*(hh_chores_num_sessions/4)* (hh_chores_num_month/12)
recode totalchoresMET2  . = 0
gen totalmowMET2 = 0
	replace totalmowMET2 = (mow_month*3.5*weight)*(mow_minutes/60)*(mow_num_session/4)* (mow_num_month/12)
recode totalmowMET2 . = 0
gen totalgolfMET2 = 0
	replace totalgolfMET2 = (golf_month*3.5*weight)*(golf_minutes/60)*(golf_num_session/4)* (golf_num_month/12)
recode totalgolfMET2 . = 0
gen totalMET2 = totalwalkMET2 + totalgardenMET2 + totalgenexerMET2 + totalmowMET2 + totalgolfMET2 + totalchoresMET2
gen posphyact2 = .
	replace posphyact2 = 1 if gender == 1 & totalMET2 < 105
	replace posphyact2 = 0 if gender == 1 & totalMET2 >= 105
	replace posphyact2 = 1 if gender == 0 & totalMET2 < 148
	replace posphyact2 = 0 if gender == 0 & totalMET2 >= 148
recode posphyact2 . = 0
gen frailty = .
	replace frailty = poswtloss + posexhaust + posgrip + posgait + posphyact2
gen nonfrail = .
	replace nonfrail = 1 if frailty == 0
	replace nonfrail = 0 if frailty >0
gen prefrail = .
	replace prefrail = 1 if frailty >0 & frailty <3
	replace prefrail = 0 if frailty == 0 | frailty >=3
gen frail = .
	replace frail = 1 if frailty >= 3
	replace frail = 0 if frailty <3
gen preorfrail = .
	replace preorfrail = 1 if frailty > 0 & frailty !=.
	replace preorfrail = 0 if frailty == 0

	*SPPB
gen sbys_stand2 = .
	replace sbys_stand2 = sbys_stand
	replace sbys_stand2 = 10 if sbys_stand >= 10 & sbys_stand !=.
gen sem_tand_stand2 = .
	replace sem_tand_stand2 = sem_tand_stand
	replace sem_tand_stand2 = 10 if sem_tand_stand >= 10 & sem_tand_stand !=.
gen tand_stand2 = .
	replace tand_stand2 = tand_stand
	replace tand_stand2 = 10 if tand_stand >= 10 & tand_stand != .
gen balancescore = .
	replace balancescore = 4 if tand_stand >= 10 & tand_stand !=.
	replace balancescore = 3 if sem_tand_stand >= 10 & sem_tand_stand !=. & tand_stand > 2 & tand_stand < 10
	replace balancescore = 2 if sem_tand_stand >= 10 & sem_tand_stand !=. & tand_stand >= 0 & tand_stand <= 2
	replace balancescore = 1 if sbys_stand >= 10 & sbys_stand !=. & sem_tand_stand < 10
	replace balancescore = 0 if sbys_stand < 10
gen invbalancescore = -balancescore
gen balancetime = .
	replace balancetime = sbys_stand2 + sem_tand_stand2 + tand_stand2
gen invbalancetime = -balancetime
gen badbalanceobj = .
	replace badbalanceobj = 1 if balancescore <= 2
	replace badbalanceobj = 0 if balancescore >=3 & balancescore != .
gen balancediscordsubjbad = .
	replace balancediscordsubjbad = 1 if balance_difficulty == 1 & badbalanceobj == 0 
	replace balancediscordsubjbad = 0 if balance_difficulty == 1 & badbalanceobj == 1
	replace balancediscordsubjbad = 0 if balance_difficulty == 0
gen balancediscord = .
	replace balancediscord = 1 if balance_difficulty == 1 & badbalanceobj == 0 
	replace balancediscord = 1 if balance_difficulty == 0 & badbalanceobj == 1
	replace balancediscord = 0 if balance_difficulty == 1 & badbalanceobj == 1
	replace balancediscord = 0 if balance_difficulty == 0 & badbalanceobj == 0

*AGS Falls
gen posAGSfallsc = .
	replace posAGSfallsc = 1 if balance_difficulty == 1 | two_falls_in_12_mo == 1
	replace posAGSfallsc = 0 if balance_difficulty == 0 & two_falls_in_12_mo == 0 & balance_difficulty != . & two_falls_in_12_mo != .

*VES-13
gen poorhealth = .
	replace poorhealth = 1 if health <= 2
	replace poorhealth = 0 if health >= 3 & health != .
gen stoopingdiff = .
	replace stoopingdiff = 1 if stooping >= 2 & stooping != .
	replace stoopingdiff = 0 if stooping <= 1
gen liftingdiff = .
	replace liftingdiff = 1 if lifting >= 2 & lifting != .
	replace liftingdiff = 0 if lifting <= 1
gen reachingdiff = .
	replace reachingdiff = 1 if reaching >= 2 & reaching != .
	replace reachingdiff = 0 if reaching <= 1
gen writingdiff = .
	replace writingdiff = 1 if writing >= 2 & writing != .
	replace writingdiff = 0 if writing <= 1
gen walkingdiff = .
	replace walkingdiff = 1 if walking >= 2 & walking != .
	replace walkingdiff = 0 if walking <= 1
gen heavyhousediff = .
	replace heavyhousediff = 1 if heavyhouse >= 2 & heavyhouse != .
	replace heavyhousediff = 0 if heavyhouse <= 1
gen vesscore1 = .
	replace vesscore1 = stoopingdiff + liftingdiff + reachingdiff + writingdiff + walkingdiff + heavyhousediff
gen vesscore1b = .
	replace vesscore1b = 1 if vesscore1 == 1
	replace vesscore1b = 2 if vesscore1 > 1 & vesscore1 != .
	replace vesscore1b = 0 if vesscore1 == 0
gen shopdiff = .
	replace shopdiff = 1 if shop >= 1 & shop != .
	replace shopdiff = 0 if shop < 1
gen manage_moneydiff = .
	replace manage_moneydiff = 1 if manage_money >= 1 & manage_money != .
	replace manage_moneydiff = 0 if manage_money < 1
gen walkroomdiff = .
	replace walkroomdiff = 1 if walkroom >= 1 & walkroom != .
	replace walkroomdiff = 0 if walkroom < 1
gen lighthousediff = .
	replace lighthousediff = 1 if lighthouse >= 1 & lighthouse != .
	replace lighthousediff = 0 if lighthouse < 1
gen bathediff = .
	replace bathediff = 1 if bathe >= 1 & bathe != .
	replace bathediff = 0 if bathe < 1
gen vesscore2 = .
	replace vesscore2 = 4 if shopdiff == 1 | manage_moneydiff == 1 | walkroomdiff == 1 | lighthousediff == 1 | bathediff == 1
	replace vesscore2 = 0 if shopdiff == 0 & manage_moneydiff == 0 & walkroomdiff == 0 & lighthousediff == 0 & bathediff == 0
gen vesagescore = .
	replace vesagescore = 1 if ves_age == 1
	replace vesagescore = 3 if ves_age == 2
	replace vesagescore = 0 if ves_age == 0
gen vestotalscore = .
	replace vestotalscore = vesagescore + poorhealth + vesscore1b + vesscore2
gen fairpoor = .
	replace fairpoor = 1 if health <= 2
	replace fairpoor = 0 if health > 2 & health !=.

*MOCA SCORING
*Calculate MoCA Blind
gen moca_score_cal_blind = repeated_digits_forward + repeated_digits_backward + finger_tap_score + serial_sevens_score + sentence_john + sentence_cats_dogs + fluency_score + train_bicycle + watch_ruler + face + velvet + church + daisy + red + date + month + year + day + place + city

*Calculate MoCA
gen moca_score_cal = trails_score + cube_score + clock_contour + clock_numbers + clock_hands + naming_lion + naming_rhino + naming_camel + repeated_digits_forward + repeated_digits_backward + finger_tap_score + serial_sevens_score + sentence_john + sentence_cats_dogs + fluency_score + train_bicycle + watch_ruler + face + velvet + church + daisy + red + date + month + year + day + place + city

*Converting MoCA Blind to MoCA scoring
gen mocablindconv = (moca_score_cal_blind*30)/22
	replace moca_score_cal = mocablindconv if participant_number==282
	replace moca_score_cal = mocablindconv if participant_number==170 & redcap_event_name=="12_month_followup_arm_1"
	replace moca_score_cal = mocablindconv if participant_number==222 & redcap_event_name=="baseline_arm_1"

*Charlsons Comorbidity Index
gen chron_pulm_dis_tx = .
	replace chron_pulm_dis_tx = 1 if chron_pulm_dis == 1 & chron_pulm_dis_med == 1
	replace chron_pulm_dis_tx = 0 if chron_pulm_dis == 0
	replace chron_pulm_dis_tx = 0 if chron_pulm_dis == 1 & chron_pulm_dis_med == 0
gen liver_dis_sev = .
	replace liver_dis_sev = 0 if liver_dis == 0
	replace liver_dis_sev = 1 if liver_dis == 1 & liver_dis_spec == 0
	replace liver_dis_sev = 3 if liver_dis == 1 & liver_dis_spec == 1
gen diabetes_sev = .
	replace diabetes_sev = 0 if diabetes == 0
	replace diabetes_sev = 1 if diabetes == 1 & diabetes_spec == 0
	replace diabetes_sev = 2 if diabetes == 1 & diabetes_spec == 1
gen renal_dis_sev = 2 if renal_dis == 1
	replace renal_dis_sev = 0 if renal_dis == 0
gen leukemia_sev = 2 if leukemia == 1
	replace leukemia_sev = 0 if leukemia == 0
gen lymphoma_sev = 2 if lymphoma == 1
	replace lymphoma_sev = 0 if lymphoma == 0
gen cancer_sev = .
	replace cancer_sev = 0 if cancer == 0
	replace cancer_sev = 2 if cancer == 1 & cancer_metastatic == 0
	replace cancer_sev = 6 if cancer == 1 & cancer_metastatic == 1
	replace cancer_sev = 8 if cancer == 1 & cancer_metastatic == 2
gen cereb_vasc_dis_sev = .
	replace cereb_vasc_dis_sev = 0 if cereb_vasc_dis == 0
	replace cereb_vasc_dis_sev = 1 if cereb_vasc_dis == 1 & cereb_vasc_dis_dif == 0
	replace cereb_vasc_dis_sev = 3 if cereb_vasc_dis == 1 & cereb_vasc_dis_dif == 1
gen charlson_cal = heart_attack + congestive + periph_vasc_dis + cereb_vasc_dis_sev + chron_pulm_dis_tx + pep_ulc_dis + liver_dis_sev + diabetes_sev + renal_dis_sev + con_tis_dis + alzheimers + leukemia_sev + lymphoma_sev + cancer_sev + aids
gen charlson_cal_inc = charlson_cal
	replace charlson_cal_inc = 50 if charlson_cal==.
label define charlson_cal_inc 50 "Missing"

*Multimorbidity
gen multimorbid = 1 if charlson_cal >=3 & charlson_cal !=.
	replace multimorbid = 0 if charlson_cal <3
	replace multimorbid = 3 if charlson_cal==.
gen multimorbid2 = 1 if charlson_cal >=2 & charlson_cal !=.
	replace multimorbid2 = 0 if charlson_cal <2

*Medications / Polypharmacy
gen fivemeds = .
	replace fivemeds = 1 if baselinemeds >=5 & baselinemeds !=.
	replace fivemeds = 0 if baselinemeds <5
gen baselinemeds_inc = baselinemeds
	replace baselinemeds_inc = 20 if baselinemeds==.
label define baselinemeds_inc 20 "Missing"
label val baselinemeds_inc baselinemeds_inc
gen medcat = .
	replace medcat = 0 if baselinemeds_inc < 5
	replace medcat = 1 if baselinemeds_inc >=5 & baselinemeds_inc !=20 & !missing(baselinemeds_inc)
	replace medcat = 2 if baselinemeds_inc==20
label val medcat baselinemeds_inc

*Create new var names for med_classes:
gen beta_blocker = .
	replace beta_blocker = 1 if med_classes___1 == 1
	replace beta_blocker = 0 if med_classes___1 == 0
gen non_beta_hyperten = .
	replace non_beta_hyperten = 1 if med_classes___2 == 1
	replace non_beta_hyperten = 0 if med_classes___2 == 0
gen anti_cholinergic = .
	replace anti_cholinergic = 1 if med_classes___3 == 1
	replace anti_cholinergic = 0 if med_classes___3 == 0
gen opioids = .
	replace opioids = 1 if med_classes___4 == 1
	replace opioids = 0 if med_classes___4 == 0
gen anti_psych = .
	replace anti_psych = 1 if med_classes___5 == 1
	replace anti_psych = 0 if med_classes___5 == 0
gen anti_depress = .
	replace anti_depress = 1 if med_classes___6 == 1
	replace anti_depress = 0 if med_classes___6 == 0
gen benzo = .
	replace benzo = 1 if med_classes___7 == 1
	replace benzo = 0 if med_classes___7 == 0
gen cholesterol_low = .
	replace cholesterol_low = 1 if med_classes___8 == 1
	replace cholesterol_low = 0 if med_classes___8 == 0
gen nsaids = .
	replace nsaids = 1 if med_classes___9 == 1
	replace nsaids = 0 if med_classes___9 == 0
gen ca_block_non_dihydro = .
	replace ca_block_non_dihydro = 1 if med_classes___10 == 1
	replace ca_block_non_dihydro = 0 if med_classes___10 == 0
gen ca_block_dihydro = .
	replace ca_block_dihydro = 1 if med_classes___11 == 1
	replace ca_block_dihydro = 0 if med_classes___11 == 0
gen anti_coag = .
	replace anti_coag = 1 if med_classes___12 == 1
	replace anti_coag = 0 if med_classes___12 == 0
gen prednisone = .
	replace prednisone = 1 if med_classes___13 == 1
	replace prednisone = 0 if med_classes___13 == 0


