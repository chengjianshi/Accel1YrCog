//Merge FACE Aging RedCap Data with DEXA and RMR Data


merge 1:1 participant_number using "Dexa Data.dta", gen(merge_2)

merge 1:1 participant_number using "RMR Data.dta", gen(merge_3)

save "Combined dataset.dta", replace
