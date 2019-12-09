"""
This script contains all functions used to process nacc and csf rows.
"""

import pandas as pd
import numpy as np
import pickle


def check_csv_columns_values(csv_file):
    """
    Gets all possible values for each column.
    Args:
        - csv_file: the csv file we want to check
    """ 

    # print out all possible values for each column
    for index, col in enumerate(csv_file.columns):
        values = pd.unique(csv_file[col])
        print(index, col)
        if type(values[-1]) != str:
            values.sort()
        print(values)
        print("total number:",len(values))
        print("\n")


def drop_columns(csv_file, file_type="nacc"):
    """
    Drop columns from nacc and csf csv
    Args:
        - csv_file: the csv file waiting for the column dropping
        - file_type: "nacc" or "csf"
    Returns:
        - csv_file_dropped: csv file after dropping some columns     
    """
    
    if file_type == "csf":
        drop_columns = ["NACCADC", "CSFABMDX", "CSFPTMDX", "CSFTTMDX"]
    
    elif file_type == "nacc": 
        # columns only containing texts  
        text_cols = ["HISPORX",  "PRIMLANX", "CVOTHRX",  "NCOTHRX", 
                     "ARTHTYPX", "OTHSLEEX", "ABUSX",    "PSYCDISX", 
                     "NPHISOX",  "COGMODEX", "BEMODEX",  "COGOTHRX", 
                     "BEOTHRX",  "MOMODEX", "NACCCGFX", "NACCBEFX",
                     "MMSELANX", "NPSYLANX", "MOCALANX", "OTHBIOMX", 
                     "OTHMUTX",  "FTLDSUBX", "OTHCOGX",  "OTHPSYX", 
                     "COGOTHX",  "COGOTH2X", "COGOTH3X", "NPFIXX", 
                     "NPTANX",   "NPABANX", "NPASANX",  "NPTDPANX", 
                     "NPPATHOX", "NACCWRI1", "NACCWRI2", "NACCWRI3", 
                     "NPFAUT1", "NPFAUT2", "NPFAUT3", "NPFAUT4",
                     "NPIQINFX"]     
        
        # administrative columns in uds and np
        admin = ["NACCADC", "PACKET", "FORMVER", "NACCVNUM", 
                 "NACCAVST", "NACCNVST", "NACCDAYS", "NACCFDYS", 
                 "NACCACTV", "NACCNOVS", "NACCDSMO", "NACCDSDY", 
                 "NACCDSYR", "BIRTHMO", "BIRTHYR", "NPFORMVER", 
                 "NPSEX", "INRACE" ]
        
        # if the patient are currently using anti-Alzheimer or anti-Parkinson medicine at the visit
        A4 = ["NACCADMD", "NACCPDMD"]

        # Subject Health History
        A5 = ["PD", "PDOTHR"]

        # CDR score, only keep CDRSUM
        B4 = ["MEMORY", "ORIENT", "JUDGMENT", "COMMUN", "HOMEHOBB", 
              "PERSCARE", "CDRGLOB", "COMPORT", "CDRLANG"]
        
        # there is a summary score in B6, so drop other columns from which the total score can be derived
        B6 = ["NOGDS", "SATIS", "DROPACT", "EMPTY", "BORED", "SPIRITS", 
              "AFRAID", "HAPPY", "HELPLESS", "STAYHOME", "MEMPROB", "WONDRFUL", 
              "WRTHLESS", "ENERGY", "HOPELESS", "BETTER"]
        
        # they are clinican judgement of symptoms
        B9 = ["B9CHG", "DECSUB", "DECIN", "DECCLIN", "DECCLCOG", "COGMEM", "COGORI", 
              "COGJUDG", "COGLANG", "COGVIS", "COGATTN", "COGFLUC", "COGFLAGO", "COGOTHR", 
              "NACCCOGF", "COGMODE", "DECAGE", "BEAPATHY", "BEDEP", "BEVHALL", "BEVWELL", 
              "BEVHAGO", "BEAHALL", "BEDEL", "BEDISIN", "BEIRRIT", "BEAGIT", "BEPERCH", "BEREM", 
              "BEREMAGO", "BEANX", "BEOTHR", "NACCBEHF", "BEMODE", "BEAGE", "DECCLMOT", "MOGAIT", 
              "MOFALLS", "MOTREM", "MOSLOW", "NACCMOTF", "MOMODE", "MOMODEX", "MOMOPARK", 
              "PARKAGE", "MOMOALS", "MOAGE", "COURSE", "FRSTCHG", "LBDEVAL", "FTLDEVAL", 
              "DECCLBE", "ALSAGE"]
        
        # MMSE Score, only keep NACCMMSE
        C1 = ["MMSECOMP", "LOGIMO", "LOGIDAY", "LOGIYR", "MMSELOC", "MMSELAN", "MMSEVIS", 
              "MMSEHEAR", "MMSEORDA", "MMSEORLO", "PENTAGON", "NPSYCLOC", "NPSYLAN", "COGSTAT", 
              "NACCC1", "LOGIPREV"]
        
        # only keep MOCATOTS for MOCA score
        C2 = ["MOCACOMP", "MOCAREAS", "MOCALOC", "MOCALAN", "MOCAVIS", "MOCAHEAR", "MOCATRAI", 
              "MOCACUBE", "MOCACLOC", "MOCACLON", "MOCACLOH", "MOCANAMI", "MOCADIGI", "MOCAREGI", 
              "MOCALETT", "MOCASER7", "MOCAREPE", "MOCAFLUE", "MOCAABST", "MOCARECN", "MOCARECC", 
              "MOCARECR", "MOCAORDT", "MOCAORMO", "MOCAORYR", "MOCAORDY", "MOCAORPL", "MOCAORCT", 
              "NACCC2"]
        
        # clinical diagonosis
        D1 = ["WHODIDDX", "DXMETHOD", "NORMCOG", "AMNDEM", "PCA", "NACCPPA", "NACCPPAG", "NACCPPME", 
              "NACCBVFT", "NACCLBDS", "NAMNDEM", "NACCTMCI", "NACCMCIL", "NACCMCIA", "NACCMCIE", 
              "NACCMCIV", "AMYLPET", "AMYLCSF", "FDGAD", "HIPPATR", "TAUPETAD", "CSFTAU", "FDGFTLD", 
              "TPETFTLD", "MRFTLD", "DATSCAN", "OTHBIOM", "IMAGLINF", "IMAGLAC", "IMAGMACH", "IMAGMICH", 
              "IMAGMWMH", "IMAGEWMH", "OTHMUT", "NACCLBDE", "NACCLBDP", "PARK", "MSAIF", "PSP", "PSPIF", 
              "CORT", "CORTIF", "FTLDMOIF", "FTLDNOS", "FTLDNOIF", "FTD", "FTDIF", "PPAPH", "PPAPHIF", 
              "FTLDSUBT", "CVD", "CVDIF", "PREVSTK", "STROKDEC", "STKIMAG", "INFNETW", "INFWMH", "VASC", 
              "VASCIF", "VASCPS", "VASCPSIF", "STROKE", "STROKIF", "ESSTREM", "ESSTREIF", "DOWNS", "DOWNSIF", 
              "HUNT", "HUNTIF", "PRION", "PRIONIF", "BRNINJ", "BRNINJIF", "HYCEPH", "HYCEPHIF", "EPILEP", 
              "EPILEPIF", "NEOP", "NEOPIF", "NEOPSTAT", "HIVIF", "OTHCOG", "OTHCOGIF", "DEP", "DEPIF", 
              "DEPTREAT", "BIPOLDIF", "SCHIZOIF", "ANXIET", "ANXIETIF", "DELIR", "DELIRIF", "PTSDDXIF", 
              "OTHPSY", "OTHPSYIF", "ALCDEM", "ALCDEMIF", "ALCABUSE", "IMPSUBIF", "DYSILL", "DYSILLIF", 
              "MEDS", "MEDSIF", "DEMUN", "DEMUNIF", "COGOTH", "COGOTHIF", "COGOTH2", "COGOTH2F", "COGOTH3", 
              "COGOTH3F", "NACCETPR", "NACCADMU", "NACCFTDM", "NACCNORM", "NACCIDEM", "DEMENTED", "NACCMCII", 
              "IMPNOMCI", "NACCALZD", "NACCALZP", "PROBAD", "PROBADIF", "POSSAD", "POSSADIF", "NACCUDSD", 
              "FTLDMO", "BRNINCTE", "HIV", "BIPOLDX", "SCHIZOP", "PTSDDX", "IMPSUB", "MSA"]
        
        # other data available in NACC 
        other = ["NACCAUTP", "NACCACSF", "NACCTCSF", "NACCPCSF"]
        
        # columns containing years information SINCE this year can indicate whether this patient has some disease
        year = ["NACCSTYR", "NACCTIYR", "PDYR", "PDOTHRYR", "HATTYEAR", "TBIYEAR"]  
        
        # all NP columns
        neuropathology = ["NPPMIH", "NPFIX", "NPWBRWT", "NPWBRF", "NACCBRNN", "NPGRHA", "NPGRSNH",
                          "NPGRLCH", "NACCAVAS", "NPTAN", "NPABAN", "NPTDPAN", "NPHISMB", "NPHISG",
                          "NPHISSS", "NPHIST", "NPHISO", "NPTHAL", "NACCBRAA", "NACCNEUR", "NPADNC",
                          "NACCDIFF", "NACCVASC", "NACCAMY", "NPLINF", "NPLAC", "NPINF", "NPINF1A",
                          "NPINF1B", "NPINF1D", "NPINF2A", "NPINF2B", "NPINF2D", "NPINF2F", "NPINF3A",
                          "NPINF3B", "NPINF3D", "NPINF3F", "NPINF4A", "NPINF4B", "NPINF4D", "NPINF4F",
                          "NACCINF", "NPHEM", "NPHEMO", "NPHEMO1", "NPMICRO", "NPOLD", "NPOLD1", 
                          "NPOLD2", "NPOLD3", "NPOLD4", "NACCMICR", "NPOLDD", "NPOLDD1", "NPOLDD2",
                          "NPOLDD3", "NPOLDD4", "NACCHEM", "NACCARTE", "NPPATH", "NPWMR", "NACCNEC",
                          "NPPATH2", "NPPATH3", "NPPATH4", "NPPATH5", "NPPATH6", "NPPATH7", "NPPATH8",
                          "NPPATH9", "NPPATH10", "NPPATH11", "NPPATHO", "NPART", "NPOANG", "NPFTDTAU",
                          "NACCPICK", "NPFTDT2", "NACCCBD", "NACCPROG", "NPFTDT5", "NPFTDT6", "NPFTDT7",
                          "NPFTDT8", "NPFTDT9", "NPFTDT10", "NPFRONT", "NPTAU", "NPFTD", "NPFTDTDP",
                          "NPALSMND", "NPOFTD1", "NPOFTD2", "NPOFTD3", "NPOFTD4", "NPOFTD5", "NPFTDNO",
                          "NPFTDSPC", "NPTDPA", "NPTDPB", "NPTDPC", "NPTDPD", "NPTDPE", "NPPDXA",
                          "NPPDXB", "NACCPRIO", "NPPDXD", "NPPDXE", "NPPDXF", "NPPDXG", "NPPDXH",
                          "NPPDXI", "NPPDXJ", "NPPDXK", "NPPDXL", "NPPDXM", "NPPDXN", "NACCDOWN", 
                          "NPPDXP", "NPPDXQ", "NACCOTHP", "NACCBNKF", "NPBNKB", "NACCFORM", "NACCPARA",
                          "NACCCSFP", "NPBNKF", "NPFAUT", "NACCDAGE", "NACCINT", "NACCLEWY", "NPLBOD",
                          "NPNLOSS", "NPHIPSCL", "NPSCL", "NPNIT", "NPCERAD", "NPADRDA", "NPOCRIT",
                          "NPVOTH", "NPLEWYCS", "NPGENE", "NPFHSPEC", "NPTAUHAP", "NPPRNP",
                          "NPCHROM", "NPPNORM", "NPCNORM", "NPPADP", "NPCADP", "NPPAD", "NPCAD", 
                          "NPPLEWY", "NPCLEWY", "NPPVASC", "NPPFTLD", "NPPHIPP", "NPCHIPP", "NPPPRION", 
                          "NPCPRION", "NPPOTH1", "NPCOTH1", "NPOTH1X", "NPPOTH2", "NPCOTH2", "NPOTH2X",
                          "NPPOTH3", "NPCOTH3", "NPOTH3X", "NPASAN", "NPGRLA", "NPGRCCA", "NPINF1F", 
                          "NPHEMO2", "NPHEMO3", "NPOFTD", "NPCVASC", "NPCFTLD"]
        # Dead information
        dead_info = ["NACCDIED", "NACCYOD", "NACCMOD"]                 
        
        drop_columns = text_cols + admin + A4 + A5 + B4 + B6 + B9 + C1 + C2 + D1 + other + year + neuropathology + dead_info + ["label"]                 
    
    else:
        raise ValueError("The file type is undefined!!")

    # drop the corresponding columns
    csv_file_dropped = csv_file.drop(columns = drop_columns)

    return csv_file_dropped


def combine_nacc_csf(nacc_csv, csf_csv):
    """
    Combine NACC and CSF together
    Args:
        - nacc_csv: csv file for nacc
        - csf_csv: csv file for csf
    Returns:
        - nacc_csv: nacc csv with csf columns    
    """    
    # get fields of csf 
    csf_cols_except_id = list(csf_csv.columns)
    csf_cols_except_id.remove("NACCID")
    csf_cols_except_id.remove("DATE")
    
    # add csf columns into nacc columns
    for col in csf_cols_except_id:
        nacc_csv[col] = np.nan

    # change the order of columns
    nacc_csv_cols = list(nacc_csv.columns)
    nacc_csv_cols.remove("NACCID")
    nacc_csv_cols.remove("DATE")
    nacc_csv_cols_reordered = ["NACCID", "DATE"] + nacc_csv_cols
    nacc_csv = nacc_csv[nacc_csv_cols_reordered]  

    # sort the nacc_csf according to NACCID and DATES
    nacc_csv = nacc_csv.sort_values(by=["NACCID", "DATE"])
    # re-index the rows
    nacc_csv=nacc_csv.reset_index(drop=True)  

    # now, we find the closet date for each records in the nacc data
    for patient in pd.unique(nacc_csv["NACCID"]):
        # Note that the NACCID field of csf is the SUBSET of nacc!!!!
        if patient not in pd.unique(csf_csv["NACCID"]):
            continue
        csf_sample = csf_csv.loc[csf_csv["NACCID"]==patient]
        nacc_sample = nacc_csv.loc[nacc_csv["NACCID"]==patient]
        for nacc_index in nacc_sample.index:
            # get the date in csf 
            nacc_time = pd.Timestamp(nacc_sample.loc[nacc_index, "DATE"])
    
            min_time_diff = 100 * 365
            min_time_index = 0
            # get the date difference between each csf_time and that nacc_time
            for csf_index in csf_sample.index:
                csf_time = pd.Timestamp(csf_sample.loc[csf_index, "DATE"])
                time_diff = abs((csf_time - nacc_time).days)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    min_time_index = csf_index
        
            # to get the closest and assign the value to it 
            nacc_csv.loc[nacc_index, csf_cols_except_id] = csf_sample.loc[min_time_index, csf_cols_except_id]
            
    return nacc_csv


def preprocess_missing_data(nacc_csf_csv):
    """
    Preprocess missing data according to categorical or continuous data. 
    Args:
        - nacc_csf_csv: combined nacc and csf rows
    Retuens:
        - nacc_csf_csv_preprocessed_missing: nacc csf rows after missing data being preprocessed
    """
    ### for continuous data, we convert missing data as NaNs in case that others influence the statisitics ###
    continuous_missing_dict = {"EDUC": [99], "INEDUC": [-4, 99], "SMOKYRS": [-4, 88, 99], 
                               "QUITSMOK": [-4, 888, 999], "BPSYS": [-4, 888], "BPDIAS": [-4, 888], 
                               "NACCGDS": [-4, 88], "NACCMMSE": [-4, 88, 95, 96, 97, 98], "LOGIMEM": [-4, 95, 96, 97, 98],
                               "MEMUNITS": [-4, 95, 96, 97, 98], "MEMTIME":[-4, 99], "UDSBENTC": [-4, 95, 96, 97, 98],
                               "UDSBENTD": [-4, 95, 96, 97, 98], "DIGIF": [-4, 95, 96, 97, 98], "DIGIFLEN": [-4, -95, 96, 97, 98], 
                               "DIGIB": [-4, 95, 96, 97, 98], "DIGIBLEN":[-4, 95, 96, 97, 98], "ANIMALS": [-4, 95, 96, 97, 98], 
                               "VEG": [-4, 95, 96, 97, 98], "TRAILA": [-4, 995, 996, 997, 998], "TRAILARR": [-4, 95, 96, 97, 98], 
                               "TRAILALI": [-4, 95, 96, 97, 98], "TRAILB":[-4, 995, 996, 997, 998], "TRAILBRR": [-4, 95, 96, 97, 98], 
                               "TRAILBLI": [-4, 95, 96, 97, 98], "WAIS": [-4, 95, 96, 97, 98], "BOSTON": [-4, 95, 96, 97, 98], 
                               "UDSVERFC": [-4, 95, 96, 97, 98], "UDSVERFN": [-4, 95, 96, 97, 98], "UDSVERNF": [-4, 95, 96, 97, 98], 
                               "UDSVERLC":[-4, 95, 96, 97, 98], "UDSVERLR":[-4, 95, 96, 97, 98], "UDSVERLN":[-4, 95, 96, 97, 98], 
                               "UDSVERTN": [-4, 95, 96, 97, 98], "UDSVERTE": [-4, 95, 96, 97, 98], "UDSVERTI": [-4, 95, 96, 97, 98], 
                               "MOCATOTS": [-4, 88], "CRAFTVRS": [-4, 95, 96, 97, 98], "CRAFTURS": [-4, 95, 96, 97, 98], 
                               "DIGFORCT": [-4, 95, 96, 97, 98], "DIGFORSL": [-4, 95, 96, 97, 98], "DIGBACCT": [-4, 95, 96, 97, 98], 
                               "DIGBACLS": [-4, 95, 96, 97, 98], "CRAFTDVR": [-4, 95, 96, 97, 98], "CRAFTDRE": [-4, 95, 96, 97, 98], 
                               "CRAFTDTI": [-4, 95, 96, 97, 98], "MINTTOTS": [-4, 95, 96, 97, 98], "MINTTOTW": [-4, 95, 96, 97, 98], 
                               "MINTSCNG": [-4, 95, 96, 97, 98], "MINTSCNC": [-4, 95, 96, 97, 98], "MINTPCNG": [-4, 95, 96, 97, 98], 
                               "MINTPCNC": [-4, 95, 96, 97, 98], "NACCAMD": [-4], "NACCBMI": [-4, 888.8]}
    
    for col, values in continuous_missing_dict.items():
        nacc_csf_csv.loc[nacc_csf_csv[col].isin(values), col] = np.nan   
    ### combine missing and unknown data together in categorical data ###
    categorical_missing_dict = {"HISPANIC": [9], "HISPOR": [-4, 88, 99], "PRIMLANG": [8, 9],
                                "INSEX": [-4], "NACCNINR": [-4, 99], "INRELTO": [-4],
                                "NACCFAM": [-4, 9], "ANYMEDS": [-4], "TOBAC30": [-4, 9],
                                "TOBAC100": [-4, 9], "PACKSPER": [-4, 8, 9], "ALCOCCAS": [-4, 9],
                                "ALCFREQ": [-4, 8, 9], "CVHATT": [-4, 9], "HATTMULT": [-4, 8, 9],
                                "CVAFIB": [-4, 9], "CVANGIO": [-4, 9], "CVBYPASS":[-4, 9], 
                                "CVPACDEF": [-4, 9], "CVPACE": [-4, 9], "CVCHF": [-4, 9], 
                                "CVANGINA": [-4, 9], "CVHVALVE": [-4, 9], "CVOTHR": [-4, 9],
                                "CBSTROKE":[-4, 9], "STROKMUL": [-4, 8, 9], "CBTIA": [-4, 9],
                                "TIAMULT": [-4, 8, 9], "SEIZURES": [-4, 9], "NACCTBI": [-4, 9],
                                "TBI": [-4, 9], "TRAUMBRF": [-4, 9], "TBIEXTEN": [-4, 9],
                                "TRAUMEXT": [-4, 9], "TBIWOLOS": [-4, 9], "TRAUMCHR": [-4, 9],
                                "NCOTHR":[-4, 9], "DIABETES": [-4, 9], "DIABTYPE": [-4, 8, 9],
                                "HYPERTEN": [-4, 9], "HYPERCHO": [-4, 9], "B12DEF": [-4, 9],
                                "THYROID": [-4, 9], "ARTHRIT": [-4, 9], "ARTHTYPE": [-4, 8, 9],
                                "ARTHUPEX": [-4, 8], "ARTHLOEX": [-4, 8], "ARTHSPIN": [-4, 8],
                                "ARTHUNK": [-4, 8], "INCONTU": [-4, 9], "INCONTF": [-4, 9],
                                "APNEA": [-4, 9], "RBD": [-4, 9], "INSOMN": [-4, 9],
                                "OTHSLEEP": [-4, 9], "ALCOHOL": [-4, 9], "ABUSOTHR": [-4, 9],
                                "PTSD": [-4, 9], "BIPOLAR": [-4, 9], "SCHIZ": [-4, 9],
                                "DEP2YRS": [-4, 9], "DEPOTHR": [-4, 9], "ANXIETY": [-4, 9],
                                "OCD": [-4, 9], "NPSYDEV": [-4, 9], "PSYCDIS": [-4, 9],
                                "NPIQINF": [-4], "DEL": [-4, 9], "DELSEV": [-4, 8, 9],
                                "HALL": [-4, 9], "HALLSEV": [-4, 8, 9], "AGIT": [-4, 9],
                                "AGITSEV": [-4, 8, 9], "DEPD": [-4, 9], "DEPDSEV": [-4, 8, 9],
                                "ANX": [-4, 9], "ANXSEV": [-4, 8, 9], "ELAT": [-4, 9],
                                "ELATSEV": [-4, 8, 9], "APA": [-4, 9], "APASEV": [-4, 8, 9],
                                "DISN": [-4, 9], "DISNSEV": [-4, 8, 9], "IRR": [-4, 9],
                                "IRRSEV": [-4, 8, 9], "MOT": [-4, 9], "MOTSEV": [-4, 8, 9],
                                "NITE": [-4, 9], "NITESEV": [-4, 8, 9], "APP": [-4, 9], 
                                "APPSEV": [-4, 8, 9], "BILLS": [-4, 8, 9], "TAXES": [-4, 8, 9],
                                "SHOPPING": [-4, 8, 9], "GAMES": [-4, 8, 9], "STOVE": [-4, 8, 9],
                                "MEALPREP": [-4, 8, 9], "EVENTS": [-4, 8, 9], "PAYATTN": [-4, 8, 9],
                                "REMDATES": [-4, 8, 9], "TRAVEL": [-4, 8, 9], "UDSBENRS": [-4, 9], 
                                "CRAFTCUE": [-4], "NACCNIHR": [99], "NACCAAAS": [-4], 
                                "NACCAANX": [-4], "NACCAC":[-4], "NACCACEI": [-4],
                                "NACCADEP": [-4], "NACCAHTN": [-4], "NACCANGI": [-4], 
                                "NACCAPSY": [-4], "NACCBETA": [-4], "NACCCCBS": [-4],
                                "NACCDBMD": [-4], "NACCDIUR": [-4], "NACCEMD": [-4],
                                "NACCEPMD": [-4], "NACCHTNC": [-4], "NACCLIPL": [-4], 
                                "NACCNSD": [-4], "NACCVASD": [-4], "NACCNE4S": [9],
                                "NACCAPOE": [9], "TBIBRIEF": [-4, 9]}

    for col, values in categorical_missing_dict.items():
        nacc_csf_csv.loc[nacc_csf_csv[col].isin(values), col] = -4
    
    # change NaNs in categorical data of original CSF csv columns with -4.
    for col in ["CSFABMD", "CSFPTMD", "CSFTTMD"]:    
        nacc_csf_csv.loc[pd.isnull(nacc_csf_csv[col]), col] = -4

    return nacc_csf_csv                                                            


def fill_continuous_missing_data_each_patient(nacc_csf_csv, fill_cols):
    """
    Fill continuous missing data using the statistics within EACH patient
    Args:
        - nacc_csf_csv: preprocessed nacc csf rows
        - fill_cols: python list storing all continuous column names
    Returns:
        - nacc_csf_csv_fill_missing_each_patient: after filled missing data    
    """

    ### for the continuous data, for each patient compute median of each PATIENT and fill NaNs ###
    for naccid in pd.unique(nacc_csf_csv["NACCID"]):
        # get the statistics for each patient
        patient_statistics = nacc_csf_csv[fill_cols].loc[nacc_csf_csv["NACCID"]==naccid].describe(include="all")
        # get the filling values
        fill_values = patient_statistics.loc["50%"].values
        fill_values_dict = dict(zip(fill_cols, fill_values))
        # fill the values for this patient
        nacc_csf_csv.loc[nacc_csf_csv["NACCID"] == naccid] = nacc_csf_csv.loc[nacc_csf_csv["NACCID"] == naccid].fillna(value=fill_values_dict)

    return nacc_csf_csv


def fill_continuous_missing_data_all_paitents(nacc_csf_csv, fill_cols, train_patient_list, valid_patient_list, test_patient_list):
    """
    Fill continuous missing data using the training median of ALL patients.
    Args:
        - nacc_csf_csv: rows after filled with each patient statistics
        - fill_cols: python list containing all continuous column names
        - train_patient_list: python list containing all NACCID in training set
        - valid_patient_list: python list containing all NACCID in validation set
        - test_patient_list: python list containing all NACCID in testing set
    Returns:
        - train_csv_filled: rows of training set after filling all the missing data
        - valid_csv_filled: rows of validation set after filling all the missing data
        - test_csv_filled: rows of testing set after filling all the missing data    
    """
    # split the nacc_csv into three subsets
    train_csv = nacc_csf_csv.loc[nacc_csf_csv["NACCID"].isin(train_patient_list)]
    valid_csv = nacc_csf_csv.loc[nacc_csf_csv["NACCID"].isin(valid_patient_list)]
    test_csv = nacc_csf_csv.loc[nacc_csf_csv["NACCID"].isin(test_patient_list)]

    # to fill NaNs in training dataset using statistics over ALL patients in the training dataset
    # to get the dictionary containing filling values
    median_statistics = train_csv[fill_cols].describe(include="all").loc["50%"]
    fill_values_dict_train_csv = dict(zip(fill_cols, median_statistics.values))
    # fill it
    train_csv_filled = train_csv.fillna(fill_values_dict_train_csv)
    
    # get the statistics of training dataset after filling values and fill test and valid set
    statistics_train_csv = train_csv_filled[fill_cols].describe(include="all").loc["50%"]
    valid_csv_filled = valid_csv.fillna(dict(zip(fill_cols, statistics_train_csv.values)))
    test_csv_filled = test_csv.fillna(dict(zip(fill_cols, statistics_train_csv.values)))

    return train_csv_filled, valid_csv_filled, test_csv_filled
    

def fill_missing_data(nacc_csf_csv_combined, fill_cols, train_patient_list, valid_patient_list, test_patient_list):
    """
    Fill the missing data for each columns of combined nacc_csf_csv file.
    Args:
        - nacc_csf_csv_combined: combined nacc and csf rows
        - fill_cols: list containing all continuous column names
        - train_paitent_list: list containing all NACCID in training dataset
        - valid_paitent_list: list containing all NACCID in validation dataset
        - test_paitent_list: list containing all NACCID in testing dataset 
    Returns:
        - train_csv_filled: rows of training set after filling all the missing data
        - valid_csv_filled: rows of validation set after filling all the missing data
        - test_csv_filled: rows of testing set after filling all the missing data
    """
    ### preprocess missing data ###
    nacc_csf_csv_preprocessed_missing = preprocess_missing_data(nacc_csf_csv_combined)
    
    ### fill continuous missing data using statistics of EACH patient ###    
    nacc_csf_csv_fill_with_each_patient = fill_continuous_missing_data_each_patient(nacc_csf_csv_preprocessed_missing, fill_cols)
    
    ### fill continuous missing data using statisics of ALL patients in training set ###
    train_csv_filled, valid_csv_filled, test_csv_filled = fill_continuous_missing_data_all_paitents( nacc_csf_csv_fill_with_each_patient, 
                                                          fill_cols, train_patient_list, valid_patient_list, test_patient_list )
    
    return train_csv_filled, valid_csv_filled, test_csv_filled

    
if __name__ == "__main__":
    
    ### check all the column values of nacc and csf csv files
    data_path = "./processed_data/all_visits_features/"
    # read all train, valid and test
    with open(data_path + "train_all_visits_features.csv", "rt") as fin:
        train_csv = pd.read_csv(fin, low_memory=False)
    with open(data_path + "valid_all_visits_features.csv", "rt") as fin:
        valid_csv = pd.read_csv(fin, low_memory=False)
    with open(data_path + "test_all_visits_features.csv", "rt") as fin:
        test_csv = pd.read_csv(fin, low_memory=False)
    # concatenate them together
    nacc_csv = pd.concat([train_csv, valid_csv, test_csv])
    #check_csv_columns_values(nacc_csv)
    # read in csf file
    with open("../data/data_with_date/csf_csv_with_dates.csv", "rt") as fin:
        csf_csv = pd.read_csv(fin, low_memory=False)
    #check_csv_columns_values(csf_csv)

    
    ### drop the columns for nacc and csf csv files ###
    nacc_csv_dropped = drop_columns(nacc_csv, "nacc")
    csf_csv_dropped = drop_columns(csf_csv, "csf")
    #check_csv_columns_values(nacc_csv_dropped)
    #check_csv_columns_values(csf_csv_dropped)

    
    ### combine NACC and CSF together ###
    nacc_csf_csv_combined = combine_nacc_csf(nacc_csv_dropped, csf_csv_dropped)
    #nacc_csv_combined.to_csv("./important_output/nacc_csv_combined.csv", index=False)
    #check_csv_columns_values(nacc_csv_combined)

    ### fill the missing data ###
    # load in continuous column names
    with open("./important_output/continuous_column_names.pkl", "rb") as fin:
        fill_column_names = pickle.load(fin)
    
    # fill the missing data for all three sets
    train_csv_filled, valid_csv_filled, test_csv_filled = fill_missing_data( nacc_csf_csv_combined, 
                                                                             fill_column_names, 
                                                                             list(pd.unique(train_csv["NACCID"])), 
                                                                             list(pd.unique(valid_csv["NACCID"])), 
                                                                             list(pd.unique(test_csv["NACCID"])) )

    # to test whether there are still some columns containing NaNs
    #print(train_csv_filled[fill_column_names].describe(include="all").loc["count"])
    #print(valid_csv_filled[fill_column_names].describe(include="all").loc["count"])
    #print(test_csv_filled[fill_column_names].describe(include="all").loc["count"])

    # write them into csv files
    train_csv_filled.to_csv("./intermediate_files/train_csv_filled.csv", index=False)
    valid_csv_filled.to_csv("./intermediate_files/valid_csv_filled.csv", index=False)
    test_csv_filled.to_csv("./intermediate_files/test_csv_filled.csv", index=False)
    
                                                                               
    
    
    
    
    
    
    
        
    