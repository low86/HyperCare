######################################
# 2026-2-1 #
# Task functions #
######################################
from tqdm import *
from pyhealth.data import  Patient
from torch.utils.data import  Dataset
from pyhealth.medcode import CrossMap
mapping = CrossMap("ICD10CM", "CCSCM")
mapping3 = CrossMap("ICD9CM", "CCSCM")


def diag_prediction_mimic4_fn(patient: Patient):

    samples = []
    visit_ls = list(patient.visits.values())
    visit_ls = sorted(visit_ls, key=lambda v: v.encounter_time)  
    for i, visit in enumerate(visit_ls):

        conditions = visit.get_code_list(table="diagnoses_icd")  
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        drugs = [drug[:4] for drug in drugs] 

        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        samples.append({
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": conditions,   
            "procedures": procedures,
            "adm_time": visit.encounter_time.strftime("%Y-%m-%d %H:%M"),
            "drugs": drugs,
            "cond_hist": conditions,  
        })

    if len(samples) < 2:
        return []

    samples[0]["cond_hist"] = [samples[0]["cond_hist"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    samples[0]["adm_time"] = [samples[0]["adm_time"]]
    samples[0]["visit_history"] = [samples[0]["visit_id"]]

    for i in range(1, len(samples)):
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["cond_hist"] = samples[i - 1]["cond_hist"] + [samples[i]["cond_hist"]]
        samples[i]["adm_time"] = samples[i - 1]["adm_time"] + [samples[i]["adm_time"]]
        samples[i]["visit_history"] = samples[i - 1]["visit_history"] + [samples[i]["visit_id"]]

    for i in range(len(samples)):
        samples[i]["cond_hist"][i] = []

    return samples


def diag_prediction_mimic3_fn(patient: Patient):
    samples = []

    visit_ls = list(patient.visits.values())
    visit_ls = sorted(visit_ls, key=lambda v: v.encounter_time)
    for i, visit in enumerate(visit_ls):
        
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        drugs = [drug[:4] for drug in drugs]  # ATC 3 level


        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "adm_time": visit.encounter_time.strftime("%Y-%m-%d %H:%M"),
                "drugs": drugs,
                "cond_hist": conditions,
            }
        )
    if len(samples) < 2:
        return []

    samples[0]["cond_hist"] = [samples[0]["cond_hist"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    samples[0]["adm_time"] = [samples[0]["adm_time"]]  
    samples[0]["visit_history"] = [samples[0]["visit_id"]]

    for i in range(1, len(samples)):
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["cond_hist"] = samples[i - 1]["cond_hist"] + [samples[i]["cond_hist"]]
        samples[i]["adm_time"] = samples[i - 1]["adm_time"] + [samples[i]["adm_time"]]
        samples[i]["visit_history"] = samples[i - 1]["visit_history"] + [samples[i]["visit_id"]]

    for i in range(len(samples)):
        samples[i]["cond_hist"][i] = []

    return samples


def drug_recommendation_mimic3_fn(patient: Patient):

    samples = []
    visit_ls = list(patient.visits.keys())
    for i in range(len(visit_ls)):
        visit = patient.visits[visit_ls[i]]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "adm_time": visit.encounter_time.strftime("%Y-%m-%d %H:%M"),
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["adm_time"] = [samples[0]["adm_time"]]  
    samples[0]["visit_history"] = [samples[0]["visit_id"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["adm_time"] = samples[i - 1]["adm_time"] + [samples[i]["adm_time"]]
        samples[i]["visit_history"] = samples[i - 1]["visit_history"] + [samples[i]["visit_id"]]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = []

    return samples


def drug_recommendation_mimic4_fn(patient: Patient):
    
    samples = []
    visit_ls = list(patient.visits.values())
    visit_ls = sorted(visit_ls, key=lambda v: v.encounter_time)
    for i, visit in enumerate(visit_ls):
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "adm_time": visit.encounter_time.strftime("%Y-%m-%d %H:%M"),
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["adm_time"] = [samples[0]["adm_time"]]  
    samples[0]["visit_history"] = [samples[0]["visit_id"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["adm_time"] = samples[i - 1]["adm_time"] + [samples[i]["adm_time"]]
        samples[i]["visit_history"] = samples[i - 1]["visit_history"] + [samples[i]["visit_id"]]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = []

    return samples