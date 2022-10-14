import matplotlib.pyplot as plt
import numpy as np

name_graph = "Precision-Recall Curve"
name_graph_2 = " - AR and Scale on Larger Features"

all__0  =[1.0, 1.0, 1.0, 1.0, 1.0, 0.9942528735632185, 0.9942528735632185, 0.9942528735632185, 0.9942528735632185, 0.9942528735632185, 0.9942528735632185, 0.9386973180076629, 0.9386973180076629, 0.9386973180076629, 0.9344941956882256, 0.9344941956882256, 0.9323962516733602, 0.9323962516733602, 0.9323962516733602, 0.9292929292929294, 0.9250269687162892, 0.9250269687162892, 0.9250269687162892, 0.9234234234234234, 0.9234234234234234, 0.9173611111111111, 0.8895833333333334, 0.8895833333333334, 0.8895833333333334, 0.8895833333333334, 0.8895833333333334, 0.6395833333333333, 0.6395833333333333, 0.6395833333333333, 0.6385542168674698, 0.6385542168674698, 0.6368715083798883, 0.6368715083798883, 0.6360360360360361, 0.6345811051693405, 0.6317460317460317, 0.6317460317460317, 0.6317460317460317, 0.6317460317460317, 0.6296296296296297, 0.6281481481481482, 0.6271929824561403, 0.6266094420600858, 0.623789764868603, 0.6234817813765182, 0.6218708827404479, 0.6205128205128205, 0.6150061500615006, 0.6141636141636142, 0.6111111111111112, 0.6052332195676905, 0.595560063105702, 0.5940371238329055, 0.589268449887206, 0.5773460531981848, 0.5773460531981848, 0.5736400601927448, 0.5705853035541849, 0.5645653735632185, 0.5583554376657824, 0.5503775620280474, 0.5444444444444444, 0.5419973544973545, 0.5417027417027417, 0.536900698302211, 0.5263179898825202, 0.5175415170667538, 0.5065221974717891, 0.4976767676767677, 0.4966269841269841, 0.49428958575300036, 0.48361998361998354, 0.4592851592851593, 0.4479486764785164, 0.3819501472357015, 0.37063364329995235, 0.20168067226890754, 0.19623655913978497, 0.15913978494623657, 0.15913978494623657, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ball_0  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.75, 0.75, 0.75, 0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
hand_0  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9666666666666667, 0.9523809523809523, 0.9523809523809523, 0.9523809523809523, 0.9402985074626866, 0.9402985074626866, 0.9402985074626866, 0.927536231884058, 0.9166666666666666, 0.9166666666666666, 0.9166666666666666, 0.9054054054054054, 0.8717948717948718, 0.8518518518518519, 0.7446808510638298, 0.7395833333333334, 0.6050420168067226, 0.5887096774193549, 0.4774193548387097, 0.4774193548387097, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
fig__0  =[1.0, 1.0, 1.0, 1.0, 1.0, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9827586206896551, 0.9701492537313433, 0.9701492537313433, 0.963855421686747, 0.963855421686747, 0.963855421686747, 0.9545454545454546, 0.941747572815534, 0.941747572815534, 0.941747572815534, 0.9369369369369369, 0.9369369369369369, 0.91875, 0.91875, 0.91875, 0.91875, 0.91875, 0.91875, 0.91875, 0.91875, 0.91875, 0.9156626506024096, 0.9156626506024096, 0.9106145251396648, 0.9106145251396648, 0.9081081081081082, 0.9037433155080213, 0.8952380952380953, 0.8952380952380953, 0.8952380952380953, 0.8952380952380953, 0.8888888888888888, 0.8844444444444445, 0.881578947368421, 0.8798283261802575, 0.8713692946058091, 0.8704453441295547, 0.8656126482213439, 0.8615384615384616, 0.8450184501845018, 0.8424908424908425, 0.8333333333333334, 0.8156996587030717, 0.803921568627451, 0.7993527508090615, 0.7850467289719626, 0.7492795389048992, 0.7492795389048992, 0.7381615598885793, 0.7289972899728997, 0.7109375, 0.6923076923076923, 0.6844660194174758, 0.680952380952381, 0.6736111111111112, 0.6727272727272727, 0.6704035874439462, 0.6386554621848739, 0.6123260437375746, 0.5920303605313093, 0.5763636363636364, 0.5732142857142857, 0.5662020905923345, 0.5454545454545454, 0.5060606060606061, 0.49199417758369723, 0.40116959064327484, 0.3723175965665236, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

all__1  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.997245179063361, 0.7750229568411386, 0.7750229568411386, 0.7750229568411386, 0.7750229568411386, 0.7750229568411386, 0.7639118457300276, 0.7639118457300276, 0.7590909090909091, 0.7590909090909091, 0.7573426573426573, 0.6573426573426574, 0.6573426573426574, 0.6534216335540839, 0.6457023060796646, 0.6427145708582834, 0.6392156862745098, 0.633879781420765, 0.633879781420765, 0.6237113402061856, 0.616504854368932, 0.6132075471698113, 0.6129032258064516, 0.6103703703703703, 0.6045197740112994, 0.6019283746556474, 0.5950520833333334, 0.5921717171717171, 0.5882352941176471, 0.5756541524459613, 0.5723684210526315, 0.5679405520169851, 0.5602409638554217, 0.5568513119533528, 0.5504067312468952, 0.5504014409133925, 0.5503963249864889, 0.5451071989849618, 0.5421545667447307, 0.5385684409363827, 0.5362021857923497, 0.5323441611009917, 0.5219525465427105, 0.5198519748218856, 0.5117312863214503, 0.5107482298909231, 0.5057295489764293, 0.5035934901401758, 0.5016026569348703, 0.48692168930390495, 0.4798258298000964, 0.47536764128882397, 0.4418307548054384, 0.3101851851851852, 0.3101851851851852, 0.3101851851851852, 0.3101851851851852, 0.3022222222222222, 0.2987012987012987, 0.29218106995884774, 0.29218106995884774, 0.26666666666666666, 0.2644927536231884, 0.23492063492063495, 0.23492063492063495, 0.2336448598130841, 0.2271386430678466, 0.2271386430678466, 0.22413793103448276, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ball_1  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
hand_1  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.953125, 0.9402985074626866, 0.9402985074626866, 0.9305555555555556, 0.9305555555555556, 0.9305555555555556, 0.9305555555555556, 0.9305555555555556, 0.9066666666666666, 0.8961038961038961, 0.8765432098765432, 0.8765432098765432, 0.8, 0.7934782608695652, 0.7047619047619048, 0.7047619047619048, 0.7009345794392523, 0.6814159292035398, 0.6814159292035398, 0.6724137931034483, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
fig__1  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9917355371900827, 0.9917355371900827, 0.9917355371900827, 0.9917355371900827, 0.9917355371900827, 0.9917355371900827, 0.9917355371900827, 0.9917355371900827, 0.9772727272727273, 0.9772727272727273, 0.972027972027972, 0.972027972027972, 0.972027972027972, 0.9602649006622517, 0.9371069182389937, 0.9281437125748503, 0.9176470588235294, 0.9016393442622951, 0.9016393442622951, 0.8711340206185567, 0.8495145631067961, 0.839622641509434, 0.8387096774193549, 0.8311111111111111, 0.8135593220338984, 0.8057851239669421, 0.78515625, 0.7765151515151515, 0.7647058823529411, 0.726962457337884, 0.7171052631578947, 0.7038216560509554, 0.6807228915662651, 0.6705539358600583, 0.6676136363636364, 0.6675977653631285, 0.6675824175824175, 0.6517150395778364, 0.6428571428571429, 0.6320987654320988, 0.625, 0.6134259259259259, 0.5822510822510822, 0.5759493670886076, 0.5515873015873016, 0.5486381322957199, 0.5335820895522388, 0.5271739130434783, 0.5212014134275619, 0.5076400679117148, 0.49917898193760263, 0.48580441640378547, 0.3949367088607595, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

all__2  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9966666666666667, 0.9966666666666667, 0.9966666666666667, 0.7744444444444444, 0.7744444444444444, 0.7744444444444444, 0.7744444444444444, 0.7744444444444444, 0.7466666666666667, 0.7327777777777778, 0.7307347670250897, 0.7307347670250897, 0.7307347670250897, 0.7307347670250897, 0.7270797962648556, 0.7270797962648556, 0.7230664689893748, 0.7230664689893748, 0.6397331356560415, 0.6392608740786878, 0.6392608740786878, 0.6392608740786878, 0.6392608740786878, 0.6392608740786878, 0.6392608740786878, 0.6360141927853341, 0.6360141927853341, 0.6349384098544233, 0.6307431390687636, 0.6296214137829045, 0.6296214137829045, 0.628667373630258, 0.628667373630258, 0.6281847660644564, 0.6281847660644564, 0.6281847660644564, 0.6151858123161928, 0.6151858123161928, 0.6110942249240122, 0.6103762492651382, 0.6087301587301588, 0.6087301587301588, 0.6076060774504355, 0.6058897243107769, 0.5951619516195162, 0.5934452438049561, 0.5927758146839419, 0.5881869772998806, 0.5845231980825202, 0.581831605087419, 0.5804473304473304, 0.5717356343168182, 0.5625568797399784, 0.5521212121212121, 0.5501906640368178, 0.5458674273703176, 0.5376385041551246, 0.5081095186478782, 0.5063712882861818, 0.49796632996633, 0.49619555131366155, 0.48895455185028847, 0.4857865378673302, 0.44773965298133356, 0.4448253776890329, 0.4374080741760505, 0.4244955748513631, 0.4227746139243753, 0.3968448252413981, 0.37235090661356257, 0.32156626102115393, 0.30765952288785403, 0.2873851294903927, 0.26418777143414823, 0.23237339904006574, 0.20091330357905335, 0.1731627865805081, 0.1138353765323993, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ball_2  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
hand_2  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9583333333333334, 0.9583333333333334, 0.9583333333333334, 0.9583333333333334, 0.9583333333333334, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9347826086956522, 0.9347826086956522, 0.9347826086956522, 0.9347826086956522, 0.9347826086956522, 0.9347826086956522, 0.9347826086956522, 0.9347826086956522, 0.8979591836734694, 0.8979591836734694, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.8666666666666667, 0.8666666666666667, 0.8666666666666667, 0.8548387096774194, 0.8484848484848485, 0.8484848484848485, 0.8484848484848485, 0.8382352941176471, 0.8169014084507042, 0.7866666666666666, 0.7866666666666666, 0.7792207792207793, 0.7625, 0.6739130434782609, 0.6702127659574468, 0.6565656565656566, 0.6565656565656566, 0.6407766990291263, 0.6407766990291263, 0.5354330708661418, 0.5354330708661418, 0.518796992481203, 0.4863013698630137, 0.4863013698630137, 0.4186046511627907, 0.35960591133004927, 0.22699386503067484, 0.22699386503067484, 0.21929824561403508, 0.1919191919191919, 0.1600831600831601, 0.14689265536723164, 0.14363636363636365, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
fig__2  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.9838709677419355, 0.9838709677419355, 0.9838709677419355, 0.9838709677419355, 0.9838709677419355, 0.9838709677419355, 0.971830985915493, 0.971830985915493, 0.971830985915493, 0.9704142011834319, 0.9704142011834319, 0.9704142011834319, 0.9704142011834319, 0.9704142011834319, 0.9704142011834319, 0.9606741573033708, 0.9606741573033708, 0.9574468085106383, 0.9574468085106383, 0.9540816326530612, 0.9540816326530612, 0.9512195121951219, 0.9512195121951219, 0.9497716894977168, 0.9497716894977168, 0.9497716894977168, 0.9475982532751092, 0.9475982532751092, 0.9404255319148936, 0.9382716049382716, 0.9333333333333333, 0.9333333333333333, 0.9299610894941635, 0.924812030075188, 0.9188191881918819, 0.9136690647482014, 0.911660777385159, 0.9097222222222222, 0.9050847457627119, 0.8970099667774086, 0.8928571428571429, 0.8769716088328076, 0.8707692307692307, 0.8696969696969697, 0.863905325443787, 0.8583815028901735, 0.850415512465374, 0.850415512465374, 0.8489010989010989, 0.8373333333333334, 0.8320209973753281, 0.8260869565217391, 0.8165829145728644, 0.8077858880778589, 0.7990430622009569, 0.7934272300469484, 0.7871853546910755, 0.7820224719101123, 0.7719298245614035, 0.7574468085106383, 0.7377049180327869, 0.6959847036328872, 0.6428571428571429, 0.6006441223832528, 0.5370370370370371, 0.45584725536992843, 0.3758519961051607, 0.3415061295971979, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

all__3  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8888888888888888, 0.8888888888888888, 0.8888888888888888, 0.8850574712643677, 0.8830409356725145, 0.827485380116959, 0.827485380116959, 0.827485380116959, 0.827485380116959, 0.827485380116959, 0.7771194808858796, 0.7637861475525463, 0.758600962367361, 0.758600962367361, 0.758600962367361, 0.6477368492802658, 0.646101071809527, 0.646101071809527, 0.644768492258566, 0.6379855465221319, 0.6379855465221319, 0.636520241171404, 0.636520241171404, 0.630228178513555, 0.62937576499388, 0.62937576499388, 0.62937576499388, 0.6272296229397155, 0.6272296229397155, 0.6272296229397155, 0.6203606526094838, 0.6203606526094838, 0.6203606526094838, 0.6141592920353983, 0.61267217630854, 0.61267217630854, 0.61267217630854, 0.6103864734299517, 0.6103864734299517, 0.6103864734299517, 0.6103864734299517, 0.6103864734299517, 0.6103864734299517, 0.6103864734299517, 0.6063380281690142, 0.6056516724336793, 0.5993265993265994, 0.5992278923875015, 0.5958608890204982, 0.5924001924001924, 0.587991718426501, 0.5829064319803844, 0.5825138026224983, 0.5683938852491562, 0.5627495291902072, 0.5573921028466483, 0.5509536784741145, 0.5453431764192046, 0.538717529966413, 0.5345461447781175, 0.5169457735247209, 0.4703256605515172, 0.45797298740123854, 0.41461829600515737, 0.3856644064365955, 0.3674212854540723, 0.3572019704433497, 0.3005932790115567, 0.29327638804148876, 0.27098364598364594, 0.24410020694046483, 0.21406516071815088, 0.05944319036869827, 0.05944319036869827, 0.04965859714463067, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ball_3  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.5, 0.5, 0.5, 0.5, 0.5, 0.35294117647058826, 0.35294117647058826, 0.35294117647058826, 0.35294117647058826, 0.35294117647058826, 0.020348837209302327, 0.020348837209302327, 0.020348837209302327, 0.020348837209302327, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
hand_3  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.96, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9302325581395349, 0.9302325581395349, 0.9302325581395349, 0.9302325581395349, 0.9302325581395349, 0.9302325581395349, 0.9302325581395349, 0.9148936170212766, 0.9148936170212766, 0.9148936170212766, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8888888888888888, 0.8888888888888888, 0.8787878787878788, 0.8787878787878788, 0.8695652173913043, 0.8695652173913043, 0.8695652173913043, 0.8356164383561644, 0.8266666666666667, 0.8181818181818182, 0.8, 0.7951807228915663, 0.7951807228915663, 0.7951807228915663, 0.7613636363636364, 0.6355140186915887, 0.6160714285714286, 0.5109489051094891, 0.4329268292682927, 0.38095238095238093, 0.35960591133004927, 0.2690909090909091, 0.2690909090909091, 0.2435064935064935, 0.19791666666666666, 0.1807511737089202, 0.17832957110609482, 0.17832957110609482, 0.148975791433892, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
fig__3  =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9885057471264368, 0.9824561403508771, 0.9824561403508771, 0.9824561403508771, 0.9824561403508771, 0.9824561403508771, 0.9824561403508771, 0.9784172661870504, 0.9784172661870504, 0.9784172661870504, 0.9784172661870504, 0.9784172661870504, 0.9784172661870504, 0.9735099337748344, 0.9735099337748344, 0.9695121951219512, 0.9695121951219512, 0.9695121951219512, 0.9651162790697675, 0.9651162790697675, 0.96045197740113, 0.9578947368421052, 0.9578947368421052, 0.9578947368421052, 0.9514563106796117, 0.9514563106796117, 0.9514563106796117, 0.9461883408071748, 0.9461883408071748, 0.9461883408071748, 0.9424778761061947, 0.9380165289256198, 0.9380165289256198, 0.9380165289256198, 0.9311594202898551, 0.9311594202898551, 0.9311594202898551, 0.9311594202898551, 0.9311594202898551, 0.9311594202898551, 0.9311594202898551, 0.9190140845070423, 0.916955017301038, 0.9090909090909091, 0.9087947882736156, 0.9087947882736156, 0.8984126984126984, 0.8944099378881988, 0.879154078549849, 0.8779761904761905, 0.8695652173913043, 0.8615819209039548, 0.8539944903581267, 0.8528610354223434, 0.8408488063660478, 0.8209718670076727, 0.8084577114427861, 0.7894736842105263, 0.7754629629629629, 0.757847533632287, 0.7329059829059829, 0.7240663900414938, 0.7213114754098361, 0.712, 0.632688927943761, 0.610738255033557, 0.5694444444444444, 0.5343839541547278, 0.4614443084455324, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


x = np.arange(0, 1.01, 0.01)

plt.figure(0)
plt.plot(x, all__0, label='default - features')
plt.plot(x, all__1, label='ar - features')
plt.plot(x, all__2, label='scale - features')
plt.plot(x, all__3, label='ar - scale - features')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(name_graph + name_graph_2)
plt.legend()
plt.savefig(name_graph + name_graph_2)

plt.figure(1)
plt.plot(x, ball_0,label='default - features')
plt.plot(x, ball_1,label='ar - features')
plt.plot(x, ball_2,label='scale - features')
plt.plot(x, ball_3,label='ar - scale - features')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(name_graph + " [Ball]" + name_graph_2)
plt.legend()
plt.savefig(name_graph + " [Ball]" + name_graph_2)

plt.figure(2)
plt.plot(x, hand_0,  label='default - features')
plt.plot(x, hand_1,  label='ar - features')
plt.plot(x, hand_2,  label='scale - features')
plt.plot(x, hand_3,  label='ar - scale - features')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(name_graph + " [Hand]" + name_graph_2)
plt.legend()
plt.savefig(name_graph + " [Hand]" + name_graph_2)

plt.figure(3)
plt.plot(x, fig__0,  label='default - features')
plt.plot(x, fig__1,  label='ar - features')
plt.plot(x, fig__2,  label='scale - features')
plt.plot(x, fig__3,  label='ar - scale - features')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(name_graph + " [Figure]" + name_graph_2)
plt.legend()
plt.savefig(name_graph + " [Figure]" + name_graph_2)

