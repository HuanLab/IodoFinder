import pandas as pd
import numpy as np
import glob
import os
from pyteomics import mzml
import joblib

# configure parameters
directory = 'C:/Users/User/Tingting/2024-04-17-iodine compounds/Model_training/negative' # Specify the directory and lcms pattern
pattern = '*.mzML'  # format of raw lcms files
ft_path = "C:/Users/User/Tingting/2024-05-24-Suwannee_HA_negative_mode/2024-05-23-HA/HA_I/extract_I_in_MS2/I_neg_3MB.csv"
model_path = 'positive_model_on_all_train.pkl' # pos: 'positive_model_on_all_train.pkl' ; neg: 'negative_model_on_all_train.pkl'
ionization_model = 'P'
rt_tol = 0.2 # 0.2 minutes rt tolerance to find the ms2 spectra for corresponding features
mz_tol = 10 # 10 ppm mass error to find the ms2 spectra for corresponding features
frag_mz_tol = 0.01 # 0.01 Da to find typical fragment ion or neutral loss

# general preprocessing
tb = pd.read_csv(ft_path) # load feature table
mzml_files = glob.glob(os.path.join(directory, pattern) ) # Construct the full pattern path
sample_num = len(tb.columns) - 3
tb['highest_sample_num'] = 0
tb.index = range(len(tb))
for i in range(len(tb)):
    intensities = tb.iloc[i, 3:(3+sample_num)]
    # Find all indices of the largest value
    indices_of_max = np.argmax(np.array(intensities))
    tb.loc[i,'highest_sample_num'] = indices_of_max

tb['ms2_mz'] = ''
tb['ms2_int'] = ''

if ionization_model == 'N':
    tb['I_frag_mz'] = 0
    tb['I_frag_int'] = 0
    tb['I_containing'] = 0
else:
    I_NL = 126.904473
    HI_NL = 127.912298
    CHIN_NL = 153.915372
    CHIO_NL = 155.907213
    I2_NL = 253.808946
    ML_features = ['loss_I_mz_closest','loss_I_average_int_closest',
                    'loss_I_mz_highest','loss_I_average_int_highest', 'loss_I_num',
                    'loss_HI_mz_closest', 'loss_HI_average_int_closest',
                    'loss_HI_mz_highest', 'loss_HI_average_int_highest','loss_HI_num',
                    'loss_I2_mz_closest', 'loss_I2_average_int_closest',
                    'loss_I2_mz_highest', 'loss_I2_average_int_highest', 'loss_I2_num',
                    'loss_CHIN_mz_closest', 'loss_CHIN_average_int_closest',
                    'loss_CHIN_mz_highest', 'loss_CHIN_average_int_highest','loss_CHIN_num',
                    'loss_CHIO_mz_closest', 'loss_CHIO_average_int_closest',
                    'loss_CHIO_mz_highest', 'loss_CHIO_average_int_highest', 'loss_CHIO_num']
    tb[ML_features] = 0
    tb['loss_I3_num'] = 0

# assign MS2 and find the ML features
for sam_i in range(len(mzml_files)):

    print(f'Loading and processing {sam_i}')
    mzml_file = mzml_files[sam_i]
    ms1_index = []
    ms1_rt_v = []
    ms2_index = []
    ms2_rt_v = []
    ms2_pre_v = []

    # Open the mzML file
    with mzml.MzML(mzml_file) as spectra:
        # Iterate over spectra

        for i in range(len(spectra)):
            spectrum = spectra[i]
            if spectrum['ms level'] == 1:
                #print(f"Spectrum ID: {spectrum['index']}")
                ms1_index.insert(len(ms1_index), i)
                ms1_rt_v.insert(len(ms1_rt_v), float(spectrum['scanList']['scan'][0]['scan start time']))

                # Check if the spectrum is an MS2 spectrum
            if spectrum['ms level'] == 2:
                #print(f"Spectrum ID: {spectrum['index']}")
                ms2_index.insert(len(ms2_index), i)
                ms2_rt_v.insert(len(ms2_rt_v),  float(spectrum['scanList']['scan'][0]['scan start time']))
                ms2_pre_v.insert(len(ms2_pre_v), spectrum['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z'])

                #spectrum['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['charge state']
                # spectrum['m/z array']
                # spectrum['intensity array']
    ms2_pre_v = np.array(ms2_pre_v)
    ms2_rt_v = np.array(ms2_rt_v)
    ms2_index = np.array(ms2_index)

    # in the future, we also need to assign ms1 spectra
    ms1_rt_v = np.array(ms1_rt_v)
    ms1_index = np.array(ms1_index)

    print('Assign ms2 spectra to each features')
    row_i = tb.index[tb['highest_sample_num'] == sam_i ]
    for row in row_i:
        # Define the target value and tolerance
        target_mz = tb.loc[row, 'mz']
        target_rt = tb.loc[row, 'rt']

        # Find positions where the value is different from the target value within the specified tolerance
        final_index = []
        pre_mz_index = np.where(np.abs(ms2_pre_v - target_mz) <= target_mz * mz_tol*0.000001)[0]
        if len(pre_mz_index) != 0:
            #print(pre_mz_index)
            pre_rt_index = np.where(np.abs(ms2_rt_v[pre_mz_index] - target_rt) <= rt_tol)[0]
            if len(pre_rt_index) == 0:
                continue
            if len(pre_rt_index) > 1:
                # pick the one with closet rt
                pre_rt_index = pre_rt_index[np.argmin(np.abs(ms2_rt_v[pre_mz_index[pre_rt_index]] - target_rt))]

            final_index = ms2_index[pre_mz_index[pre_rt_index]]
        else:
            continue

        final_index = int(final_index)
        with mzml.MzML(mzml_file) as spectra:
            spectrum = spectra[final_index]
            ms2_mz = spectrum['m/z array']
            ms2_int = spectrum['intensity array']
            int_index = np.where( ms2_int > 5)[0]
            if len(int_index) == 0:
                ms2_mz = []
                ms2_int = []
            else:
                ms2_mz = ms2_mz[int_index]
                ms2_int = ms2_int[int_index]
                ms2_int = ms2_int/max(ms2_int)*100

        if len(ms2_mz) ==0:
            continue
        int_index = np.where( ms2_int >= 0.5)[0]

        ms2_mz = ms2_mz[int_index]
        ms2_int = ms2_int[int_index]
        # Convert the array to a string with elements separated by commas
        rounded_ms2_mz = np.round(ms2_mz, 4)
        rounded_ms2_int = np.round(ms2_int, 2)
        np.set_printoptions(precision=4, suppress=False, threshold=np.inf)
        ms2_mz_str = np.array2string(rounded_ms2_mz, separator=',', precision=4, suppress_small=False).strip('[]').replace(' ','')
        ms2_int_str = np.array2string(rounded_ms2_int, separator=',', precision=2, suppress_small=False).strip('[]').replace(' ','')

        tb.loc[row, 'ms2_mz'] = ms2_mz_str
        tb.loc[row, 'ms2_int'] = ms2_int_str

        if ionization_model == 'N':
            # looking for I ion fragment and its intensity with 0.01
            I_frag_mz_index = np.where(np.abs(ms2_mz - 126.905022) <= frag_mz_tol)[0]
            if len(I_frag_mz_index) == 0:
                continue
            if len(I_frag_mz_index) > 1:
                # find the one with the highest intensity
                I_frag_mz_index = I_frag_mz_index[np.argmax( ms2_int[I_frag_mz_index])]

            tb.loc[row, 'I_frag_mz'] = ms2_mz[I_frag_mz_index]
            tb.loc[row, 'I_frag_int'] = ms2_int[I_frag_mz_index]
        else:
            t_mz = tb.loc[row, 'mz']
            t_mz = t_mz.astype(float)
            # check whether they have precursors in ms2, if not, add it with 10% intenisty #
            pre_index = np.where( np.abs(ms2_mz - t_mz) <= t_mz*0.00002)[0]
            if(len(pre_index) ==0):
                ms2_mz = np.append(ms2_mz, t_mz)
                ms2_int = np.append(ms2_int, 10)

            if len(ms2_mz) ==1:
                continue
            sort_indices = np.argsort(ms2_mz)
            # Reverse the indices to get the order from largest to smallest
            descending_indices = sort_indices[::-1]
            ms2_mz = ms2_mz[descending_indices]
            ms2_int = ms2_int[descending_indices]

            difference_matrix = np.subtract.outer(ms2_mz, ms2_mz)
            # I_NL
            difference_NL = np.abs(difference_matrix - I_NL)
            NL_indices = np.where( difference_NL <= frag_mz_tol)
            NL_indice1 = NL_indices[0]
            NL_indice2 = NL_indices[1]
            if len(NL_indice1) != 0:
                print(f'we are checking NL of I ')
                tb.loc[row, 'loss_I_num'] = len(NL_indice1)

                # closest
                mz_closet_i = np.argmin(difference_NL[ NL_indices])
                closet_i1 = NL_indice1[mz_closet_i]
                closet_i2 = NL_indice2[mz_closet_i]
                mz_diff = ms2_mz[closet_i1] - ms2_mz[closet_i2]
                average_int =  ms2_int[closet_i1]*0.5 + ms2_int[closet_i2]*0.5
                tb.loc[row, 'loss_I_mz_closest'] = mz_diff
                tb.loc[row, 'loss_I_average_int_closest'] = average_int

                # highest intensity
                ms2_int_i = np.argmax( ms2_int[NL_indice1] + ms2_int[NL_indice2])

                mz_diff = ms2_mz[NL_indice1[ms2_int_i]] - ms2_mz[NL_indice2[ms2_int_i]]
                average_int =  (ms2_int[NL_indice1[ms2_int_i]] *0.5 + ms2_int[NL_indice2[ms2_int_i]]*0.5)
                tb.loc[row, 'loss_I_mz_highest'] = mz_diff
                tb.loc[row, 'loss_I_average_int_highest'] = average_int

            # HI_NL
            difference_NL = np.abs(difference_matrix - HI_NL)
            NL_indices = np.where( difference_NL <= frag_mz_tol)
            NL_indice1 = NL_indices[0]
            NL_indice2 = NL_indices[1]
            if len(NL_indice1) != 0:
                print(f'we are checking NL of HI ')
                tb.loc[row, 'loss_HI_num'] = len(NL_indice1)

                # closest
                mz_closet_i = np.argmin(difference_NL[ NL_indices])
                closet_i1 = NL_indice1[mz_closet_i]
                closet_i2 = NL_indice2[mz_closet_i]
                mz_diff = ms2_mz[closet_i1] - ms2_mz[closet_i2]
                average_int =  ms2_int[closet_i1]*0.5 + ms2_int[closet_i2]*0.5
                tb.loc[row, 'loss_HI_mz_closest'] = mz_diff
                tb.loc[row, 'loss_HI_average_int_closest'] = average_int

                # highest intensity
                ms2_int_i = np.argmax( ms2_int[NL_indice1] + ms2_int[NL_indice2])
                mz_diff = ms2_mz[NL_indice1[ms2_int_i]] - ms2_mz[NL_indice2[ms2_int_i]]
                average_int =  (ms2_int[NL_indice1[ms2_int_i]] *0.5 + ms2_int[NL_indice2[ms2_int_i]]*0.5)
                tb.loc[row, 'loss_HI_mz_highest'] = mz_diff
                tb.loc[row, 'loss_HI_average_int_highest'] = average_int

            # CHIN_NL
            difference_NL = np.abs(difference_matrix - CHIN_NL)
            NL_indices = np.where( difference_NL <= frag_mz_tol)
            NL_indice1 = NL_indices[0]
            NL_indice2 = NL_indices[1]
            if len(NL_indice1) != 0:
                print(f'we are checking NL of CHIN ')
                tb.loc[row, 'loss_CHIN_num'] = len(NL_indice1)

                # closest
                mz_closet_i = np.argmin(difference_NL[ NL_indices])
                closet_i1 = NL_indice1[mz_closet_i]
                closet_i2 = NL_indice2[mz_closet_i]
                mz_diff = ms2_mz[closet_i1] - ms2_mz[closet_i2]
                average_int =  ms2_int[closet_i1]*0.5 + ms2_int[closet_i2]*0.5
                tb.loc[row, 'loss_CHIN_mz_closest'] = mz_diff
                tb.loc[row, 'loss_CHIN_average_int_closest'] = average_int

                # highest intensity
                ms2_int_i = np.argmax( ms2_int[NL_indice1] + ms2_int[NL_indice2])

                mz_diff = ms2_mz[NL_indice1[ms2_int_i]] - ms2_mz[NL_indice2[ms2_int_i]]
                average_int =  (ms2_int[NL_indice1[ms2_int_i]] *0.5 + ms2_int[NL_indice2[ms2_int_i]]*0.5)
                tb.loc[row, 'loss_CHIN_mz_highest'] = mz_diff
                tb.loc[row, 'loss_CHIN_average_int_highest'] = average_int

            # CHIO_NL
            difference_NL = np.abs(difference_matrix - CHIO_NL)
            NL_indices = np.where( difference_NL <= frag_mz_tol)
            NL_indice1 = NL_indices[0]
            NL_indice2 = NL_indices[1]
            if len(NL_indice1) != 0:
                print(f'we are checking NL of CHINO ')
                tb.loc[row, 'loss_CHIO_num'] = len(NL_indice1)
                # closest
                mz_closet_i = np.argmin(difference_NL[ NL_indices])
                closet_i1 = NL_indice1[mz_closet_i]
                closet_i2 = NL_indice2[mz_closet_i]
                mz_diff = ms2_mz[closet_i1] - ms2_mz[closet_i2]
                average_int =  ms2_int[closet_i1]*0.5 + ms2_int[closet_i2]*0.5
                tb.loc[row, 'loss_CHIO_mz_closest'] = mz_diff
                tb.loc[row, 'loss_CHIO_average_int_closest'] = average_int

                # highest intensity
                ms2_int_i = np.argmax( ms2_int[NL_indice1] + ms2_int[NL_indice2])
                mz_diff = ms2_mz[NL_indice1[ms2_int_i]] - ms2_mz[NL_indice2[ms2_int_i]]
                average_int =  (ms2_int[NL_indice1[ms2_int_i]] *0.5 + ms2_int[NL_indice2[ms2_int_i]]*0.5)
                tb.loc[row, 'loss_CHIO_mz_highest'] = mz_diff
                tb.loc[row, 'loss_CHIO_average_int_highest'] = average_int

            # I2_NL
            difference_NL = np.abs(difference_matrix - I2_NL)
            NL_indices = np.where( difference_NL <= frag_mz_tol)
            NL_indice1 = NL_indices[0]
            NL_indice2 = NL_indices[1]
            if len(NL_indice1) != 0:
                print(f'we are checking NL of I2 ')
                tb.loc[row, 'loss_I2_num'] = len(NL_indice1)
                # closest
                mz_closet_i = np.argmin(difference_NL[ NL_indices])
                closet_i1 = NL_indice1[mz_closet_i]
                closet_i2 = NL_indice2[mz_closet_i]
                mz_diff = ms2_mz[closet_i1] - ms2_mz[closet_i2]
                average_int =  ms2_int[closet_i1]*0.5 + ms2_int[closet_i2]*0.5
                tb.loc[row, 'loss_I2_mz_closest'] = mz_diff
                tb.loc[row, 'loss_I2_average_int_closest'] = average_int

                # highest intensity
                ms2_int_i = np.argmax( ms2_int[NL_indice1] + ms2_int[NL_indice2])

                mz_diff = ms2_mz[NL_indice1[ms2_int_i]] - ms2_mz[NL_indice2[ms2_int_i]]
                average_int =  (ms2_int[NL_indice1[ms2_int_i]] *0.5 + ms2_int[NL_indice2[ms2_int_i]]*0.5)
                tb.loc[row, 'loss_I2_mz_highest'] = mz_diff
                tb.loc[row, 'loss_I2_average_int_highest'] = average_int

            # I3_NL
            difference_NL = np.abs(difference_matrix - 380.713419)
            NL_indices = np.where( difference_NL <= frag_mz_tol)
            NL_indice1 = NL_indices[0]
            NL_indice2 = NL_indices[1]
            if len(NL_indice1) != 0:
                print(f'we are checking NL of I3 ')
                tb.loc[row, 'loss_I3_num'] = len(NL_indice1)

# output the prediction results
if ionization_model == 'N':
    X = tb[['I_frag_mz','I_frag_int']]
    model = joblib.load(model_path)
    pred = model.predict(X)
    tb['prediction'] = pred
    tb.to_csv(f"negative mode {len(tb)} features, {len(tb[tb['ms2_mz'] != ''])} has ms2 spectra {len(tb[tb['I_frag_mz'] != 0])} I frag {len(tb[tb['prediction'] == 1])} I compounds.csv")
else:
    X = tb[ML_features]
    model = joblib.load(model_path)

    pred = model.predict(X)
    tb['prediction'] = pred
    tb['prediction'].value_counts()
    tb.to_csv(f"positive mode {len(tb)} features, {len(tb[tb['ms2_mz'] != ''])} has ms2 spectra {len(tb[tb['prediction'] == 1])} I compounds.csv")

# check Isotopes (C13 1.0034, Cl37 1.9971), Adduct (positive and negative)
tb = tb[tb['prediction'] == 1]
tb['isotopes']= ''
tb['adduct'] = ''
tb_sorted = tb.sort_values(by='mz', ascending=True)
tb_sorted.index = range(len(tb_sorted))
for i in range(len(tb_sorted)):
    t_mz = tb_sorted.loc[i,'mz']
    t_rt = tb_sorted.loc[i,'rt']

    rt_index = np.where( np.abs(t_rt - np.array(tb_sorted['rt'])) <= 0.05)[0]

    if(len(rt_index) == 1):
        continue
    #print(i)
    # look for M+1, 13C
    M1_index = np.where( np.abs(t_mz + 1.0034 - np.array(tb_sorted['mz'][rt_index])) <= t_mz*0.00002 )[0]

    if(len(M1_index) != 0):
        print(f' {rt_index[M1_index]} is M+1 for {i} feature')
        tb_sorted['isotopes'][ rt_index[M1_index]] = tb_sorted['isotopes'][ rt_index[M1_index]] + ';'+'M+1 for F' + str(tb_sorted['featureID'][i])

    # look for M+2, 13C
    M2_index = np.where( np.abs(t_mz + 1.0034*2 - np.array(tb_sorted['mz'][rt_index])) <= t_mz*0.00002 )[0]
    if(len(M2_index) != 0):
        print(f' {rt_index[M2_index]} is M+2 for {i} feature')
        tb_sorted['isotopes'][ rt_index[M2_index]] = tb_sorted['isotopes'][ rt_index[M2_index]] + ';'+'M+2 for F' + str(tb_sorted['featureID'][i])

    # look for M+2, Cl
    M2_index = np.where( np.abs(t_mz + 1.9971 - np.array(tb_sorted['mz'][rt_index])) <= t_mz*0.00002 )[0]

    if(len(M2_index) != 0):
        print(f' {rt_index[M2_index]} is 37Cl M+2 for {i} feature')
        tb_sorted['isotopes'][ rt_index[M2_index]] = tb_sorted['isotopes'][ rt_index[M2_index]] + ';'+'M+2 for F' + str(tb_sorted['featureID'][i])

    # look for M+4, Cl
    M4_index = np.where( np.abs(t_mz + 1.9971*2 - np.array(tb_sorted['mz'][rt_index])) <= t_mz*0.00002 )[0]

    if(len(M4_index) != 0):
        print(f' {rt_index[M4_index]} is 37Cl M+4 for {i} feature')
        tb_sorted['isotopes'][ rt_index[M4_index]] = tb_sorted['isotopes'][ rt_index[M4_index]] + ';'+ 'M+4 for F' + str(tb_sorted['featureID'][i])

    # look for adducts
    if ionization_model == 'P':
        default_adduct = '[M+H]+'
        pos_adduct_form = ( '[M+Na]+',  '[M+K]+', '[M+NH4]+', '[M+H-H2O]+', '[M+H+CH3OH]+', '[2M+H]+', '[2M+NH4]+')
        pos_adduct_mass = ( +21.98194, +37.95588,  +17.02655,   -18.010565,    +32.026215 ,  -1.007276, +16.01927)

        for n in range(len(pos_adduct_mass)):
            if pos_adduct_form[n] != '[2M+H]+':
                adduct_index = np.where( np.abs(t_mz + pos_adduct_mass[n] - np.array(tb_sorted['mz'][rt_index])) <= t_mz*0.00002 )[0]
                if len(adduct_index) != 0:
                    print(pos_adduct_form[n])
                    tb_sorted['adduct'][ rt_index[adduct_index]] = tb_sorted['adduct'][ rt_index[adduct_index]]+';'+ pos_adduct_form[n] + ' for F' + str(tb_sorted['featureID'][i])
            else:
                adduct_index = np.where( np.abs(2*t_mz + pos_adduct_mass[n] - np.array(tb_sorted['mz'][rt_index])) <= t_mz*0.00002 )[0]
                if len(adduct_index) != 0:
                    print(pos_adduct_form[n])
                    tb_sorted['adduct'][ rt_index[adduct_index]] = tb_sorted['adduct'][ rt_index[adduct_index]]+';'+pos_adduct_form[n] + ' for F' + str(tb_sorted['featureID'][i])

    else:
        default_adduct = '[M-H]-'
        neg_adduct_form = ('[M-H2O-H]-', '[2M-H]-', '[M+Cl]-', '[M+I]-')
        neg_adduct_mass = (-18.010565, +1.007276, +35.97668, +127.9123)
        for n in range(len(neg_adduct_mass)):
            if neg_adduct_form[n] != '[2M-H]-':
                adduct_index = np.where( np.abs(t_mz + neg_adduct_mass[n] - np.array(tb_sorted['mz'][rt_index])) <= t_mz*0.00002 )[0]
                if len(adduct_index) != 0:
                    print(neg_adduct_form[n])
                    tb_sorted['adduct'][ rt_index[adduct_index]] = tb_sorted['adduct'][ rt_index[adduct_index]]+ ';'+ neg_adduct_form[n] + ' for F' + str(tb_sorted['featureID'][i])
            else:
                adduct_index = np.where( np.abs(2*t_mz + neg_adduct_mass[n] - np.array(tb_sorted['mz'][rt_index])) <= t_mz*0.00002 )[0]
                if len(adduct_index) != 0:
                    print(neg_adduct_form[n])
                    tb_sorted['adduct'][ rt_index[adduct_index]] = tb_sorted['adduct'][ rt_index[adduct_index]]+';'+neg_adduct_form[n] + ' for F' + str(tb_sorted['featureID'][i])

isotope_num = len(tb_sorted[tb_sorted['isotopes'] != ''])
adduct_num = len(tb_sorted[tb_sorted['adduct'] != ''])
tb_sorted.to_csv(f'{len(tb_sorted)} iodine chemicals {isotope_num} isotopes {adduct_num} adducts.csv')
