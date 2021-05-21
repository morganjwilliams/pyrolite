import numpy as np
import pandas as pd
import scipy
import periodictable as pt
from .mindb import get_mineral_group, list_minerals, parse_composition
from ..comp.codata import renormalise, close
from ..geochem.transform import convert_chemistry, to_molecular
from ..util.log import Handle

logger = Handle(__name__)


def unmix(comp, parts, ord=1, det_lim=0.0001):
    """
    From a composition and endmember components, find a set of weights which best
    approximate the composition as a weighted sum of components.

    Parameters
    --------------
    comp : :class:`numpy.ndarray`
        Array of compositions (shape :math:`n_S, n_C`).
    parts : :class:`numpy.ndarray`
        Array of endmembers (shape :math:`n_E, n_C`).
    ord : :class:`int`
        Order of regularization, defaults to L1 for sparsity.
    det_lim : :class:`float`
        Detection limit, below which minor components will be omitted for sparsity.

    Returns
    --------
    :class:`numpy.ndarray`
        Array of endmember modal abundances (shape :math:`n_S, n_E`)
    """
    nsamples, nscomponents = comp.shape
    nparts, ncomponents = parts.shape
    assert nscomponents == ncomponents
    weights = np.ones((nsamples, nparts))
    weights /= weights.sum()
    bounds = np.array([np.zeros(weights.size), np.ones(weights.size)]).T

    def fn(x, comp, parts):
        x = x.reshape(nsamples, nparts)
        return np.linalg.norm(x.dot(parts) - comp, ord=ord)

    res = scipy.optimize.minimize(
        fn,
        weights.flatten(),
        bounds=bounds,
        args=(comp, parts),
        constraints={"type": "eq", "fun": lambda x: np.sum(x) - nsamples},
    )
    byparts = res.x.reshape(weights.shape)
    byparts[(np.isclose(byparts, 0.0, atol=1e-06) | (byparts <= det_lim))] = 0.0
    # if the abundances aren't already molecular, this would be the last point
    # where access access to the composition of the endmembers is guaranteed
    return close(byparts)


def CIPW_norm(data, form="weight"):
    """
    Standardised calcuation of estimated mineralogy from bulk rock chemistry.
    Takes a dataframe of chemistry & creates a dataframe of estimated mineralogy.
    
    This is the CIPW norm of Verma et al. (2003)
    
    This version only uses major elements
    Parameters
    -----------
    df : :class:`p&as.DataFrame`
        Dataframe containing compositions to transform.
    Returns
    --------
    :class:`pandas.DataFrame`
    Notes
    -----
    This function is currently a stub.
    """
    columns = ['SiO2','TiO2','Al2O3','Fe2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5','CO2','SO3']
    mol_weights = pd.Series([60.08,79.90,101.96,159.6882,71.844,70.94,40.30,56.08,61.98,94.20,141.94, 44.01, 80.06], index = columns)

    # If want to convert to FeOtotal
#     data['FeO'] = data['FeO'] + (data['Fe2O3']*(2*71.8444/159.6882))
#     data['Fe2O3'] = 0

    # Normalise to 100 on anhydrous basis
    totals = data.sum(axis = 1)
    data = data.divide(totals/100, axis = 0)

    # Convert to moles
    data = data.div(mol_weights)

    # Calculate corrected molecular weights for later
    # When updating for trace elements will need to add more here
    xFeO = data['FeO']/(data['FeO']+data['MnO'])
    xMnO = data['MnO']/(data['FeO']+data['MnO'])
    MWFeOcorr = (xMnO*mol_weights['MnO'])+(xFeO*mol_weights['FeO'])
#     MWFeOcorr = (mol_weights['MnO']*(data['MnO']/(data['MnO']+data['FeO']))+(mol_weights['FeO']*data['FeO']/(data['MnO']+data['FeO'])))
    MWil = MWFeOcorr + 79.8658
    MWhyp = MWFeOcorr + 60.0843
    MWhyp = (100.39*(data['MgO']/(data['FeO']+data['MgO'])))+(MWhyp*(data['FeO']/(data['FeO']+data['MgO'])))
    MWol = (2*MWFeOcorr) + 60.0843
    MWol = (140.69*(data['MgO']/(data['FeO']+data['MgO'])))+(MWol*(data['FeO']/(data['FeO']+data['MgO'])))
    # MWhyp = (55.845+24.305)+(((28.085)+(16*3))*2)
    MWmt = (MWFeOcorr) + 159.6882


    Fe_mol_weights = pd.DataFrame(list(zip(MWil,MWhyp, MWol, MWmt)),
                                  columns = ['ilmenite', 'hypersthene', 'olivine','magnetite'])


    # Add MnO to FeO
    data['FeO'] = data['FeO'] + data['MnO']
    data = data.drop(['MnO'], axis = 1)

    # Calculate molecular weights for some more difficult minerals
    MWAp = 3*mol_weights['CaO']+(1/3)*(mol_weights['CaO']-15.9994)+154.6101241 # apatite
    MWNaCl = ((mol_weights['Na2O']-15.9994)/2) + 35.4527 # Halite - adding MW Cl to Na
    MWCaF2 = (mol_weights['CaO']-15.9994) + (2*18.9984032) # Fluorite - add fluorine to Ca MWR
    MWFeS2 = (mol_weights['FeO']-15.9994) + (2*32.066) # Pyrite - add sulphur to Fe

    # Normative minerals
    #Zircon??
    # Apatite
    data['apatite'] = np.where(data['CaO']>=((3+(1/3))*data['P2O5']), data['P2O5'], data['CaO']/(3+(1/3)))
    P2O5 = np.where(data['CaO']<((3+(1/3))*data['P2O5']), data['P2O5']-data['apatite'], 0)
    data['CaO'] = np.where(data['CaO']>=((3+(1/3))*data['P2O5']), data['CaO']-((3+(1/3))*data['apatite']), 0)
    data['P2O5'] = P2O5
    # Fluorite
    # Halite
    # Thenardite
    data['thenardite'] = np.where(data['SO3'] != 0, np.where(data['Na2O']>=data['SO3'], data['SO3'],data['Na2O']), 0)
    Na2O = np.where(data['SO3'] != 0, np.where(data['Na2O']>=data['SO3'], data['Na2O']-data['thenardite'], data['Na2O']), data['Na2O'])
    data['SO3'] = np.where(data['SO3'] != 0, np.where(data['Na2O']<data['SO3'], data['SO3']-data['thenardite'], 0), 0)
    data['Na2O'] = Na2O
    # Pyrite
    # Carbonate/calcite
    # Chromite
    # Ilmenite
    data['ilmenite'] = np.where(data['FeO']>=data['TiO2'], data['TiO2'], data['FeO'])
    FeO = np.where(data['FeO']>=data['TiO2'], data['FeO']-data['ilmenite'], 0)
    data['TiO2'] = np.where(data['FeO']<data['TiO2'], data['TiO2']-data['ilmenite'], 0)
    data['FeO'] = FeO
    # Orthoclase
    data['orthoclase'] = np.where(data['Al2O3']>=data['K2O'], data['K2O'], data['Al2O3'])
    Al2O3 = np.where(data['Al2O3']>=data['K2O'], data['Al2O3']-data['orthoclase'], 0)
    data['K2O'] = np.where(data['Al2O3']<data['K2O'], data['K2O']-data['orthoclase'], 0)
    data['ks'] = data['K2O']
    Y = (data['orthoclase']*6)+data['ks']
    data['Al2O3'] = Al2O3

    # Albite
    data['albite'] = np.where(data['Al2O3']>=data['Na2O'], data['Na2O'], data['Al2O3'])
    Al2O3 = np.where(data['Al2O3']>=data['Na2O'],data['Al2O3']-data['albite'], 0)
    data['Na2O'] = np.where(data['Al2O3']<data['Na2O'], data['Na2O']-data['albite'], 0)
    data['Al2O3'] = Al2O3
    Y = Y + 6*data['albite']

    # Acmite - for Fe2O3
    data['acmite'] = np.where(data['Na2O']>=data['Fe2O3'], data['Fe2O3'], data['Na2O'])
    Na2O = np.where(data['Na2O']>=data['Fe2O3'], data['Na2O']-data['acmite'], 0)
    data['ns'] = np.where(data['Na2O']>=data['Fe2O3'], Na2O, 0)
    data['Fe2O3'] = np.where(data['Na2O']>=data['Fe2O3'], 0, data['Fe2O3']-data['acmite'])
    data['Na2O'] = Na2O
    Y = Y + (4*data['acmite'])
    print(data['acmite'])

    # Anorthite
    data['anorthite'] = np.where(data['Al2O3']>=data['CaO'], data['CaO'], data['Al2O3'])
    Al2O3 = np.where(data['Al2O3']>=data['CaO'], data['Al2O3']-data['anorthite'], 0)
    Y = np.where(data['Al2O3']>=data['CaO'], 2*data['anorthite']+Y, Y)
    data['corundum'] = np.where(data['Al2O3']>=data['CaO'], Al2O3, 0)
    data['CaO'] = np.where(data['Al2O3']<data['CaO'], data['CaO']-data['anorthite'], 0)
    data['Al2O3'] = Al2O3
    Y = Y + 2*data['anorthite']


    # Sphene/rutile - not sure if working correctly
    data['titanite'] = np.where(data['CaO']>=data['TiO2'], data['TiO2'], data['CaO'])
    CaO = np.where(data['CaO']>=data['TiO2'], data['CaO']-data['titanite'], 0)
    TiO2 = np.where(data['CaO']<data['TiO2'], data['TiO2']-data['titanite'], 0)
    data['rutile'] = np.where(data['CaO']<data['TiO2'], TiO2, 0)
    Y = Y + data['titanite']
    data['TiO2'] = TiO2
    data['CaO'] = CaO

    # Magnetite for Fe2O3 & FeO
    data['magnetite'] = np.where(data['Fe2O3']>=data['FeO'], data['FeO'], data['Fe2O3'])
    Fe2O3 = np.where(data['Fe2O3']>=data['FeO'], data['Fe2O3']-data['magnetite'], 0)
    data['FeO'] = np.where(data['Fe2O3']>=data['FeO'], 0, data['FeO']-data['magnetite'])
    data['Fe2O3'] = Fe2O3

    # Subdivided normative minerals
    data['MgFeO'] = data['MgO']+data['FeO']
    data['Mg/Fe'] = data['MgO']/(data['FeO']+data['MgO'])
    data['Fe/Mg'] = data['FeO']/(data['FeO']+data['MgO'])

    # diopside,wollastonite, hypersthene
    data['diopside'] = np.where(data['CaO']>=data['MgFeO'], data['MgFeO'], data['CaO'])
    CaO = np.where(data['CaO']>=data['MgFeO'], data['CaO']-data['diopside'], 0)
    MgFeO = np.where(data['CaO']>=data['MgFeO'], 0, data['MgFeO']-data['diopside'])
    data['wollastonite'] = np.where(data['CaO']>=data['MgFeO'], CaO, 0)
    data['hypersthene'] = np.where(data['CaO']>=data['MgFeO'], 0, MgFeO)
    Y = np.where(data['CaO']>=data['MgFeO'], Y+(2*data['diopside'])+data['wollastonite'], Y+(2*data['diopside'])+data['hypersthene'])
    data['CaO'] = CaO
    data['MgFeO'] = MgFeO


    # Quartz/undersaturated minerals
    data['quartz'] = np.where(data['SiO2']>=Y, data['SiO2']-Y, 0)
    D = np.where(data['SiO2']<Y, Y-data['SiO2'], 0)

    # Olivine
    olivine = np.where((D != 0) & (D<data['hypersthene']/2), D, 0)
    olivine = np.where((D != 0) & (D>=data['hypersthene']/2), data['hypersthene']/2, olivine)
    D1 = np.where((D != 0) & (D>=data['hypersthene']/2),D-(data['hypersthene']/2),0)
    hypersthene = np.where((D != 0) & (D<data['hypersthene']/2), data['hypersthene']-(2*D), data['hypersthene'])
    data['hypersthene'] = np.where((D != 0) & (D>=data['hypersthene']/2), 0, hypersthene)
    data['olivine'] = olivine

    # Sphene/perovskite
    titanite = np.where((D1 != 0) & (D1<data['titanite']), data['titanite']-D1, data['titanite'])
    titanite = np.where((D1 != 0) & (D1>=data['titanite']), 0, titanite)
    data['perovskite'] = np.where((D1 != 0) & (D1<data['titanite']), D1, 0)
    data['perovskite'] = np.where((D1 != 0) & (D1>=data['titanite']), data['titanite'], titanite)
    D2 = np.where((D1 != 0) & (D1>=data['titanite']),D1-data['titanite'],0)
    data['titanite'] = titanite

    # Nepheline
    nepheline = np.where((D2 != 0) & (D2<(4*data['albite'])), D2/4, 0)
    nepheline = np.where((D2 != 0) & (D2>=(4*data['albite'])), data['albite'], nepheline)
    albite = np.where((D2 != 0) & (D2<(4*data['albite'])), data['albite']-(D2/4), data['albite'])
    albite = np.where((D2 != 0) & (D2>=(4*data['albite'])), 0, albite)
    D3 = np.where((D2 != 0) & (D2>=(4*data['albite'])), D2-4*(data['albite']),0)
    data['albite'] = albite
    data['nepheline'] = nepheline

    # Leucite
    leucite = np.where((D3 != 0) & (D3<2*data['orthoclase']), D3/2, 0)
    leucite = np.where((D3 != 0) & (D3>=2*data['orthoclase']), data['orthoclase'], leucite)
    orthoclase = np.where((D3 != 0) & (D3<2*data['orthoclase']), data['orthoclase']-(D3/2),data['orthoclase'])
    orthoclase = np.where((D3 != 0) & (D3>=2*data['orthoclase']), 0, orthoclase)
    D4 = np.where((D3 != 0) & (D3>=2*data['orthoclase']),D3-(2*data['orthoclase']),0)
    data['leucite'] = leucite
    data['orthoclase'] = orthoclase

    # Dicalcium silicate/wollastonite
    data['cs'] = np.where((D4 != 0) & (D4<data['wollastonite']/2), D4, data['wollastonite']/2)
    data['cs'] = np.where(D4 == 0,0,data['cs'])
    wollastonite = np.where((D4 != 0) & (D4<data['wollastonite']/2), data['wollastonite']-(2*D4), data['wollastonite'])
    wollastonite = np.where((D4 != 0) & (D4>=data['wollastonite']/2), 0, wollastonite)
    D5 = np.where((D4 != 0) & (D4>=data['wollastonite']/2),D4-(data['wollastonite']/2),0)
    data['wollastonite'] = wollastonite


    #Dicalcium silicate/olivine
    olivine = np.where((D5 != 0) & (D5<data['diopside']), data['olivine']+(D5/2), data['olivine'])
    olivine = np.where((D5 != 0) & (D5>=data['diopside']), data['olivine']+(data['diopside']/2), olivine)                     
    cs = np.where((D5 != 0) & (D5<data['diopside']), data['cs']+(D5/2), data['cs'])
    data['cs'] = np.where((D5 != 0) & (D5>=data['diopside']), data['cs']+(data['diopside']/2), cs)
    D6 = np.where((D5 != 0) & (D5>=data['diopside']), D5-data['diopside'],0)                           
    diopside = np.where(D5<data['diopside'], data['diopside']-D5, data['diopside'])
    data['diopside'] = np.where(D5>=data['diopside'], 0, diopside) 
    data['olivine'] = olivine
    data['cs'] = cs

    # Kaliophilite/leucite
    kaliophilite = np.where((D6 != 0) & (data['leucite']>=(D6/2)), D6/2, 0)
    data['kaliophilite'] = np.where((D6 != 0) & (data['leucite']<(D6/2)), data['leucite'],kaliophilite)
    DEFSIO2 = np.where((D6 != 0) & (data['leucite']<(D6/2)), D6-(2*data['kaliophilite']),0)
    leucite = np.where((D6 != 0) & (data['leucite']>=(D6/2)), data['leucite']-(D6/2),data['leucite'])
    data['leucite'] = np.where((D6 != 0) & (data['leucite']<(D6/2)), 0,leucite)
    # Subdivide hy & di into Mg & Fe??

    # Convert normative minerals to % by multiplying by molecular weights
    # Will need to do Fe minerals separately
    norm = data[['apatite', 'thenardite', 'ilmenite', 'orthoclase',
           'albite', 'acmite','anorthite', 'corundum', 'titanite', 'rutile', 'magnetite',
           'diopside', 'wollastonite', 'hypersthene', 'quartz',
           'olivine', 'perovskite', 'nepheline', 'leucite', 'cs', 'kaliophilite']]
    print(norm['hypersthene'])
    mol_weights = pd.Series([MWAp, 142.04314, 1, 556.663076, 524.446016,462.00434,278.207276, 101.961276, 196.0275, 79.8658,1,
                  216.5504, 116.1617, 1, 60.0853, 1, 135.9432, 284.108816, 436.494476, 172.2391, 316.325876],
                            index = norm.columns)
    norm = norm.mul(mol_weights, axis = 1)
    norm[['ilmenite','hypersthene','olivine','magnetite']] = norm[['ilmenite','hypersthene','olivine','magnetite']].mul(Fe_mol_weights, axis = 1)
    norm['total'] = norm.sum(axis = 1)
    norm.iloc[3]
    return norm
