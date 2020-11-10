def RTPv2trainedmodel():
    from RTP_V2_Model_Scaffold import RTPv2model
    from pandas import read_csv as rc
    z000xfaexl1wbs0 = rc('000xfaexl1wbs[0].csv')
    z000xfaexl1wbs1 = rc('000xfaexl1wbs[1].csv')
    z000xfaexl2wbs0 = rc('000xfaexl2wbs[0].csv')
    RTPv2modeli = RTPv2model()

    l1wbs=[z000xfaexl1wbs0,z000xfaexl1wbs1]
    l2wbs=[z000xfaexl2wbs0]
    RTPv2modeli.layers[0].set_weights(l1wbs)
    RTPv2modeli.layers[1].set_weights(l2wbs)
    
    return RTPv2modeli
