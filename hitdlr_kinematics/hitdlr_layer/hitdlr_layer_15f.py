# hithand layer for torch
import torch
import math
import trimesh
import glob
import os
import numpy as np
import copy
# from taxonomy_20dof import grasp_dict_20f
# from utils_.taxonomy_15dof import grasp_dict_15f
# All lengths are in mm and rotations in radians
hand_base_points = [7, 10, 14, 15, 18, 20, 26, 29, 39, 42, 44, 47, 50, 51, 52, 53, 62, 63, 66, 67, 71, 73, 75, 76, 79,
                    82, 84, 86, 88, 97, 98, 99, 100, 103, 106, 108, 117, 119, 120, 124, 125, 127, 129, 130, 132, 133,
                    137, 138, 144, 145, 147, 148, 153, 155, 158, 169, 170, 175, 177, 178, 183, 184, 186, 188, 189, 192,
                    194, 195, 198, 200, 204, 213, 218, 224, 227, 228, 237, 238, 239, 243, 245, 253, 255, 257, 260, 261,
                    262, 265, 275, 277, 278, 284, 288, 304, 306, 307, 308, 310, 311, 312, 317, 318, 323, 325, 330, 331,
                    336, 337, 338, 344, 345, 352, 354, 363, 364, 368, 372, 377, 379, 390, 393, 398, 406, 408, 414, 415,
                    428, 433, 447, 448, 450, 451, 454, 456, 458, 460, 462, 476, 478, 479, 481, 483, 484, 485, 491, 497,
                    498, 506, 508, 514, 517, 518, 520, 522, 524, 531, 535, 536, 541, 548, 549, 554, 555, 560, 569, 572,
                    581, 583, 584, 585, 586, 602, 603, 609, 616, 618, 624, 626, 627, 630, 643]

# finger 1 joint 1
f_1_j_1_points = [661, 662, 665, 674, 677, 680, 681, 685, 690, 693, 699, 700, 715, 726, 731, 733, 734, 738, 742, 748,
                  750, 751, 755, 756, 757, 770, 776, 786, 788, 790, 801, 804, 809, 816, 818, 822, 836, 839, 847, 858,
                  862, 863, 868, 875, 881, 884, 886, 893, 894, 900, 913, 916, 919, 920, 921, 922, 924, 926, 928, 931,
                  935, 939, 942, 947, 950, 951, 958, 961, 965, 968, 970, 973, 974, 976, 985, 991, 998, 1000, 1001,
                  1002, 1003, 1006, 1012, 1013, 1014, 1015, 1017, 1020, 1021, 1023, 1024, 1027, 1033, 1037, 1038, 1044,
                  1058, 1065, 1067, 1069]

f_1_j_2_points = [2764, 2766, 2771, 2773, 2778, 2779, 2784, 2798, 2803, 2806, 2807, 2808, 2810,
                  2815, 2817, 2821, 2824, 2825, 2831, 2845, 2864, 2878, 2879, 2882, 2886, 2887, 2890, 2893, 2900, 2904,
                  2914, 2918, 2920, 2925, 2932, 2933, 2934, 2940, 2941, 2944, 2945, 2962, 2964, 2968, 2970, 2972, 2973,
                  2977, 2982, 2984, 2986, 2991, 2993, 3000, 3001, 3005, 3022, 3032, 3033, 3037, 3048, 3053, 3060, 3063,
                  3066, 3069, 3071]

f_1_j_3_points = [4328, 4331, 4332, 4337, 4340, 4344, 4345, 4350, 4354, 4360, 4377, 4378, 4381, 4385,
                  4388, 4390, 4391, 4393, 4395, 4396, 4397, 4399, 4400, 4404, 4407, 4409, 4424, 4427, 4429, 4431, 4432,
                  4433, 4435, 4436, 4438, 4447, 4449, 4450, 4453, 4454, 4456, 4457, 4459, 4460, 4463, 4464, 4465, 4468,
                  4470, 4471, 4477, 4478, 4482, 4484, 4487, 4490, 4491, 4493, 4494, 4495, 4498, 4500, 4504, 4505, 4509,
                  4521, 4525, 4526, 4528, 4530, 4533, 4537, 4539, 4543, 4554, 4557, 4561, 4568, 4572, 4573, 4581, 4582,
                  4587, 4589, 4594, 4595, 4598, 4601, 4602, 4603, 4604, 4605, 4607, 4608, 4612, 4616, 4619, 4620, 4625,
                  4626, 4627, 4636, 4637, 4642, 4645, 4650, 4654, 4659, 4661, 4664, 4667, 4668, 4669, 4672, 4673, 4677,
                  4678, 4679, 4684, 4685, 4691, 4694, 4698, 4711, 4713, 4720, 4722, 4725, 4727]

finger_1_points = f_1_j_1_points + f_1_j_2_points + f_1_j_3_points

f_2_j_1_points = [1083, 1084, 1087, 1096, 1099, 1102, 1103, 1107, 1112, 1115, 1121, 1122, 1137, 1148, 1153, 1155, 1156,
                  1160, 1164, 1170, 1172, 1173, 1177, 1178, 1179, 1192, 1198, 1208, 1210, 1212, 1223, 1226, 1231, 1238,
                  1240, 1244, 1258, 1261, 1269, 1280, 1284, 1285, 1290, 1297, 1303, 1306, 1308, 1315, 1316, 1322, 1335,
                  1338, 1341, 1342, 1343, 1344, 1346, 1348, 1350, 1353, 1357, 1361, 1364, 1369, 1372, 1373, 1380, 1383,
                  1387, 1390, 1392, 1395, 1396, 1398, 1407, 1413, 1420, 1422, 1423, 1424, 1425, 1428, 1434, 1435, 1436,
                  1437, 1439, 1442, 1443, 1445, 1446, 1449, 1455, 1459, 1460, 1466, 1480, 1487, 1489, 1491]

f_2_j_2_points = [3078, 3080, 3085, 3087, 3092, 3093, 3098, 3112, 3117, 3120, 3121, 3122, 3124, 3129, 3131, 3135, 3138, 3139, 3145,
                  3159, 3178, 3192, 3193, 3196, 3200, 3201, 3204, 3207, 3214, 3218, 3228, 3232, 3234, 3239, 3246, 3247,
                  3248, 3254, 3255, 3258, 3259, 3276, 3278, 3282, 3284, 3286, 3287, 3291, 3296, 3298, 3300, 3305, 3307,
                  3314, 3315, 3319, 3336, 3346, 3347, 3351, 3362, 3367, 3374, 3377, 3380, 3383, 3385]

f_2_j_3_points = [4730, 4733, 4734, 4739, 4742, 4746, 4747, 4752, 4756, 4762, 4779, 4780, 4783, 4787, 4790, 4792, 4793, 4795, 4797, 4798,
                  4799, 4801, 4802, 4806, 4809, 4811, 4826, 4829, 4831, 4833, 4834, 4835, 4837, 4838, 4840, 4849, 4851,
                  4852, 4855, 4856, 4858, 4859, 4861, 4862, 4865, 4866, 4867, 4870, 4872, 4873, 4879, 4880, 4884, 4886,
                  4889, 4892, 4893, 4895, 4896, 4897, 4900, 4902, 4906, 4907, 4911, 4923, 4927, 4928, 4930, 4932, 4935,
                  4939, 4941, 4945, 4956, 4959, 4963, 4970, 4974, 4975, 4983, 4984, 4989, 4991, 4996, 4997, 5000, 5003,
                  5004, 5005, 5006, 5007, 5009, 5010, 5014, 5018, 5021, 5022, 5027, 5028, 5029, 5038, 5039, 5044, 5047,
                  5052, 5056, 5061, 5063, 5066, 5069, 5070, 5071, 5074, 5075, 5079, 5080, 5081, 5086, 5087, 5093, 5096,
                  5100, 5113, 5115, 5122, 5124, 5127, 5129]

finger_2_points = f_2_j_1_points + f_2_j_2_points + f_2_j_3_points

f_3_j_1_points = [1505, 1506, 1509, 1518, 1521, 1524, 1525, 1529, 1534, 1537, 1543, 1544, 1559, 1570, 1575, 1577, 1578,
                  1582, 1586, 1592, 1594, 1595, 1599, 1600, 1601, 1614, 1620, 1630, 1632, 1634, 1645, 1648, 1653, 1660,
                  1662, 1666, 1680, 1683, 1691, 1702, 1706, 1707, 1712, 1719, 1725, 1728, 1730, 1737, 1738, 1744, 1757,
                  1760, 1763, 1764, 1765, 1766, 1768, 1770, 1772, 1775, 1779, 1783, 1786, 1791, 1794, 1795, 1802, 1805,
                  1809, 1812, 1814, 1817, 1818, 1820, 1829, 1835, 1842, 1844, 1845, 1846, 1847, 1850, 1856, 1857, 1858,
                  1859, 1861, 1864, 1865, 1867, 1868, 1871, 1877, 1881, 1882, 1888, 1902, 1909, 1911, 1913]

f_3_j_2_points = [3392, 3394,3399, 3401, 3406, 3407, 3412, 3426, 3431, 3434, 3435, 3436, 3438, 3443, 3445, 3449, 3452, 3453, 3459,
                  3473, 3492, 3506, 3507, 3510, 3514, 3515, 3518, 3521, 3528, 3532, 3542, 3546, 3548, 3553, 3560, 3561,
                  3562, 3568, 3569, 3572, 3573, 3590, 3592, 3596, 3598, 3600, 3601, 3605, 3610, 3612, 3614, 3619, 3621,
                  3628, 3629, 3633, 3650, 3660, 3661, 3665, 3676, 3681, 3688, 3691, 3694, 3697, 3699]

f_3_j_3_points = [5132, 5135, 5136,5141, 5144, 5148, 5149, 5154, 5158, 5164, 5181, 5182, 5185, 5189, 5192, 5194, 5195, 5197, 5199, 5200,
                  5201, 5203, 5204, 5208, 5211, 5213, 5228, 5231, 5233, 5235, 5236, 5237, 5239, 5240, 5242, 5251, 5253,
                  5254, 5257, 5258, 5260, 5261, 5263, 5264, 5267, 5268, 5269, 5272, 5274, 5275, 5281, 5282, 5286, 5288,
                  5291, 5294, 5295, 5297, 5298, 5299, 5302, 5304, 5308, 5309, 5313, 5325, 5329, 5330, 5332, 5334, 5337,
                  5341, 5343, 5347, 5358, 5361, 5365, 5372, 5376, 5377, 5385, 5386, 5391, 5393, 5398, 5399, 5402, 5405,
                  5406, 5407, 5408, 5409, 5411, 5412, 5416, 5420, 5423, 5424, 5429, 5430, 5431, 5440, 5441, 5446, 5449,
                  5454, 5458, 5463, 5465, 5468, 5471, 5472, 5473, 5476, 5477, 5481, 5482, 5483, 5488, 5489, 5495, 5498,
                  5502, 5515, 5517, 5524, 5526, 5529, 5531]

finger_3_points = f_3_j_1_points + f_3_j_2_points + f_3_j_3_points

f_4_j_1_points = [1927, 1928, 1931, 1940, 1943, 1946, 1947, 1951, 1956, 1959, 1965, 1966, 1981, 1992, 1997, 1999, 2000,
                  2004, 2008, 2014, 2016, 2017, 2021, 2022, 2023, 2036, 2042, 2052, 2054, 2056, 2067, 2070, 2075, 2082,
                  2084, 2088, 2102, 2105, 2113, 2124, 2128, 2129, 2134, 2141, 2147, 2150, 2152, 2159, 2160, 2166, 2179,
                  2182, 2185, 2186, 2187, 2188, 2190, 2192, 2194, 2197, 2201, 2205, 2208, 2213, 2216, 2217, 2224, 2227,
                  2231, 2234, 2236, 2239, 2240, 2242, 2251, 2257, 2264, 2266, 2267, 2268, 2269, 2272, 2278, 2279, 2280,
                  2281, 2283, 2286, 2287, 2289, 2290, 2293, 2299, 2303, 2304, 2310, 2324, 2331, 2333, 2335]

f_4_j_2_points = [3706, 3708, 3713, 3715, 3720, 3721, 3726, 3740, 3745, 3748, 3749, 3750, 3752, 3757, 3759, 3763, 3766, 3767, 3773,
                  3787, 3806, 3820, 3821, 3824, 3828, 3829, 3832, 3835, 3842, 3846, 3856, 3860, 3862, 3867, 3874, 3875,
                  3876, 3882, 3883, 3886, 3887, 3904, 3906, 3910, 3912, 3914, 3915, 3919, 3924, 3926, 3928, 3933, 3935,
                  3942, 3943, 3947, 3964, 3974, 3975, 3979, 3990, 3995, 4002, 4005, 4008, 4011, 4013]

f_4_j_3_points = [5534, 5537, 5538, 5543, 5546, 5550, 5551, 5556, 5560, 5566, 5583, 5584, 5587, 5591, 5594, 5596, 5597, 5599, 5601, 5602,
                  5603, 5605, 5606, 5610, 5613, 5615, 5630, 5633, 5635, 5637, 5638, 5639, 5641, 5642, 5644, 5653, 5655,
                  5656, 5659, 5660, 5662, 5663, 5665, 5666, 5669, 5670, 5671, 5674, 5676, 5677, 5683, 5684, 5688, 5690,
                  5693, 5696, 5697, 5699, 5700, 5701, 5704, 5706, 5710, 5711, 5715, 5727, 5731, 5732, 5734, 5736, 5739,
                  5743, 5745, 5749, 5760, 5763, 5767, 5774, 5778, 5779, 5787, 5788, 5793, 5795, 5800, 5801, 5804, 5807,
                  5808, 5809, 5810, 5811, 5813, 5814, 5818, 5822, 5825, 5826, 5831, 5832, 5833, 5842, 5843, 5848, 5851,
                  5856, 5860, 5865, 5867, 5870, 5873, 5874, 5875, 5878, 5879, 5883, 5884, 5885, 5890, 5891, 5897, 5900,
                  5904, 5917, 5919, 5926, 5928, 5931, 5933]

finger_4_points = f_4_j_1_points + f_4_j_2_points + f_4_j_3_points

f_5_j_1_points = [2349, 2350, 2353, 2362, 2365, 2368, 2369, 2373, 2378, 2381, 2387, 2388, 2403, 2414, 2419, 2421, 2422,
                  2426, 2430, 2436, 2438, 2439, 2443, 2444, 2445, 2458, 2464, 2474, 2476, 2478, 2489, 2492, 2497, 2504,
                  2506, 2510, 2524, 2527, 2535, 2546, 2550, 2551, 2556, 2563, 2569, 2572, 2574, 2581, 2582, 2588, 2601,
                  2604, 2607, 2608, 2609, 2610, 2612, 2614, 2616, 2619, 2623, 2627, 2630, 2635, 2638, 2639, 2646, 2649,
                  2653, 2656, 2658, 2661, 2662, 2664, 2673, 2679, 2686, 2688, 2689, 2690, 2691, 2694, 2700, 2701, 2702,
                  2703, 2705, 2708, 2709, 2711, 2712, 2715, 2721, 2725, 2726, 2732, 2746, 2753, 2755, 2757]

f_5_j_2_points = [4020, 4022, 4027, 4029, 4034, 4035, 4040, 4054, 4059, 4062, 4063, 4064, 4066, 4071, 4073, 4077, 4080, 4081, 4087,
                  4101, 4120, 4134, 4135, 4138, 4142, 4143, 4146, 4149, 4156, 4160, 4170, 4174, 4176, 4181, 4188, 4189,
                  4190, 4196, 4197, 4200, 4201, 4218, 4220, 4224, 4226, 4228, 4229, 4233, 4238, 4240, 4242, 4247, 4249,
                  4256, 4257, 4261, 4278, 4288, 4289, 4293, 4304, 4309, 4316, 4319, 4322, 4325, 4327]

f_5_j_3_points = [5936, 5939, 5940, 5945, 5948, 5952, 5953, 5958, 5962, 5968, 5985, 5986, 5989, 5993, 5996, 5998, 5999, 6001, 6003, 6004,
                  6005, 6007, 6008, 6012, 6015, 6017, 6032, 6035, 6037, 6039, 6040, 6041, 6043, 6044, 6046, 6055, 6057,
                  6058, 6061, 6062, 6064, 6065, 6067, 6068, 6071, 6072, 6073, 6076, 6078, 6079, 6085, 6086, 6090, 6092,
                  6095, 6098, 6099, 6101, 6102, 6103, 6106, 6108, 6112, 6113, 6117, 6129, 6133, 6134, 6136, 6138, 6141,
                  6145, 6147, 6151, 6162, 6165, 6169, 6176, 6180, 6181, 6189, 6190, 6195, 6197, 6202, 6203, 6206, 6209,
                  6210, 6211, 6212, 6213, 6215, 6216, 6220, 6224, 6227, 6228, 6233, 6234, 6235, 6244, 6245, 6250, 6253,
                  6258, 6262, 6267, 6269, 6272, 6275, 6276, 6277, 6280, 6281, 6285, 6286, 6287, 6292, 6293, 6299, 6302,
                  6306, 6319, 6321, 6328, 6330, 6333, 6335]

finger_5_points = f_5_j_1_points + f_5_j_2_points + f_5_j_3_points

# 1 2 3 bottom middle top
joint_1_points = f_1_j_1_points + f_2_j_1_points + f_3_j_1_points + f_4_j_1_points + f_5_j_1_points
joint_2_points = f_1_j_2_points + f_2_j_2_points + f_3_j_2_points + f_4_j_2_points + f_5_j_2_points
joint_3_points = f_1_j_3_points + f_2_j_3_points + f_3_j_3_points + f_4_j_3_points + f_5_j_3_points

TaxPoints = {
    'Parallel_Extension':   joint_2_points + joint_3_points,
    'Pen_Pinch':            f_1_j_3_points + f_2_j_3_points + f_3_j_3_points,
    'Palmar_Pinch':         f_1_j_3_points + f_1_j_2_points + f_2_j_3_points + f_2_j_2_points + f_3_j_3_points + f_3_j_2_points,
    'Precision_Sphere':     joint_1_points + joint_2_points + joint_3_points,
    'Large_Wrap':           joint_1_points + joint_2_points + joint_3_points,
}

all_points_list = [finger_1_points, finger_2_points, finger_3_points, finger_4_points, finger_5_points]
all_points = []
for part in all_points_list:
    all_points = all_points + part


def save_to_mesh(vertices, faces, output_mesh_path=None):
    assert output_mesh_path is not None
    with open(output_mesh_path, 'w') as fp:
        for vert in vertices:
            fp.write('v %f %f %f\n' % (vert[0], vert[1], vert[2]))
        for face in faces+1:
            fp.write('f %d %d %d\n' % (face[0], face[1], face[2]))
    print('Output mesh save to: ', os.path.abspath(output_mesh_path))


class HitdlrLayer(torch.nn.Module):
    def __init__(self, device='cuda'):
        # The forward kinematics equations implemented here are from
        super().__init__()
        self.A0 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.A1 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.A2 = torch.tensor(0.001 * 55, dtype=torch.float32, device=device)
        self.A3 = torch.tensor(0.001 * 25, dtype=torch.float32, device=device)
        # self.Dw = torch.tensor(0.001 * 76, dtype=torch.float32, device=device)
        # self.Dw_knuckle = torch.tensor(0.001 * 42, dtype=torch.float32, device=device)
        # self.D3 = torch.tensor(0.001 * 9.5, dtype=torch.float32, device=device)
        # self.D1 = torch.tensor(0.0, dtype=torch.float32, device=device)
        # self.D2 = torch.tensor(0.0, dtype=torch.float32, device=device)
        # self.D3 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.phi0 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.phi1 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.phi2 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.phi3 = torch.tensor(0.0, dtype=torch.float32, device=device)
        # self.phi3 = torch.tensor(-math.pi/2, dtype=torch.float32, device=device)
        dir_path = os.path.split(os.path.abspath(__file__))[0]

        self.T = torch.from_numpy(np.load(os.path.join(dir_path, './T.npy')).astype(np.float32)).to(device).reshape(-1, 4, 4)
        # self.T_AR = torch.tensor([[0.45621433, -0.54958655, -0.69987364, 0.062569057],
        #                           [0.1048023,  0.81419986,  -0.57104734, 0.044544548],
        #                           [0.88367696, 0.18717161,  0.42904758,  0.080044647],
        #                           [0,         0,         0,        1]], dtype=torch.float32, device=device)

        self.T_AR = torch.tensor([[0.429052, -0.571046, -0.699872, 0.061],
                                  [0.187171,  0.814201,  -0.549586, 0.044],
                                  [0.883675, 0.104806,  0.456218,  0.0885],
                                  [0,         0,         0,        1]], dtype=torch.float32, device=device)

        self.T_AR_normal = torch.tensor([[0.429052, -0.571046, -0.699872, 0],
                                        [0.187171,  0.814201,  -0.549586, 0],
                                        [0.883675, 0.104806,  0.456218,  0],
                                        [0,         0,         0,        1]], dtype=torch.float32, device=device)

        self.T_BR = torch.tensor([[0.0,      -0.087156,  0.996195, -0.001429881],
                                  [0.0,      -0.996195, -0.087156,  0.036800135],
                                  [1.0,       0.0,       0.0,       0.116743545],
                                  [0,         0,         0,         1]], dtype=torch.float32, device=device)

        self.T_BR_normal = torch.tensor([[0.0,      -0.087156,  0.996195, 0],
                                        [0.0,      -0.996195, -0.087156,  0],
                                        [1.0,       0.0,       0.0,       0],
                                        [0,         0,         0,         1]], dtype=torch.float32, device=device)

        self.T_CR = torch.tensor([[0.0,  0.0, 1.0, -0.0026],
                                  [0.0, -1.0, 0.0,  0.01],
                                  [1.0,  0.0, 0.0,  0.127043545],
                                  [0,    0,   0,    1]], dtype=torch.float32, device=device)

        self.T_CR_normal = torch.tensor([[0.0,  0.0, 1.0, 0],
                                        [0.0, -1.0, 0.0,  0],
                                        [1.0,  0.0, 0.0,  0],
                                        [0,    0,   0,    1]], dtype=torch.float32, device=device)

        self.T_DR = torch.tensor([[0.0,  0.087152, 0.996195, -0.001429881],
                                  [0.0, -0.996195, 0.087152, -0.016800135],
                                  [1.0,  0.0,      0.0,       0.122043545],
                                  [0,         0,         0,        1]], dtype=torch.float32, device=device)

        self.T_DR_normal = torch.tensor([[0.0,  0.087152, 0.996195, 0],
                                        [0.0, -0.996195, 0.087152, 0],
                                        [1.0,  0.0,      0.0,       0],
                                        [0,         0,         0,        1]], dtype=torch.float32, device=device)

        self.T_ER = torch.tensor([[0.0,  0.1736479, 0.9848078,  0.002071571],
                                  [0.0, -0.9848078, 0.1736479, -0.043396306],
                                  [1.0,  0.0,      0.0,       0.103043545],
                                  [0,         0,         0,        1]], dtype=torch.float32, device=device)

        self.T_ER_normal = torch.tensor([[0.0,  0.1736479, 0.9848078,  0],
                                         [0.0, -0.9848078, 0.1736479, 0],
                                         [1.0,  0.0,      0.0,       0],
                                         [0,         0,         0,   1]], dtype=torch.float32, device=device)

        # self.pi_0_5 = torch.tensor(math.pi / 2, dtype=torch.float32, device=device)
        self.device = device
        self.meshes = self.load_meshes()

        self.righthand = self.meshes["righthand_base"][0]

        self.righthand_normal = self.meshes["righthand_base"][2]

        self.base = self.meshes['base'][0]
        self.base_normal = self.meshes['base'][2]

        self.proximal = self.meshes['proximal'][0]
        self.proximal_normal = self.meshes['proximal'][2]

        self.medial = self.meshes['medial'][0]
        self.medial_normal = self.meshes['medial'][2]
        self.distal = self.meshes['distal'][0]
        self.distal_normal = self.meshes['distal'][2]

        self.gripper_faces = [
            self.meshes["righthand_base"][1],  # self.meshes["palm_2"][1],
            self.meshes['base'][1], self.meshes['proximal'][1],
            self.meshes['medial'][1], self.meshes['distal'][1]
        ]

        # self.vertice_face_areas = [
        #     self.meshes["righthand_base"][2],  # self.meshes["palm_2"][2],
        #     self.meshes['base'][2], self.meshes['proximal'][2],
        #     self.meshes['medial'][2], self.meshes['distal'][2]
        # ]

        # self.num_vertices_per_part = [
        #     self.meshes["righthand_base"][0].shape[0],  # self.meshes["palm_2"][0].shape[0],
        #     self.meshes['base'][0].shape[0], self.meshes['proximal'][0].shape[0],
        #     self.meshes['medial'][0].shape[0], self.meshes['distal'][0].shape[0]
        # ]

    def load_meshes(self):
        # mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/../meshes/hitdlr_hand_coarse/*"
        mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/../meshes/hitdlr_hand_tmp/*"
        mesh_files = glob.glob(mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}
        for mesh_file in mesh_files:
            name = os.path.basename(mesh_file)[:-4]
            mesh = trimesh.load(mesh_file)
            # triangle_areas = trimesh.triangles.area(mesh.triangles)
            # vert_area_weight = []
            # for i in range(mesh.vertices.shape[0]):
            #     vert_neighour_face = np.where(mesh.faces == i)[0]
            #     vert_area_weight.append(1000000*triangle_areas[vert_neighour_face].mean())
            temp = torch.ones(mesh.vertices.shape[0], 1).float()
            vertex_normals = copy.deepcopy(mesh.vertex_normals)
            meshes[name] = [
                torch.cat((torch.FloatTensor(np.array(mesh.vertices)), temp), dim=-1).to(self.device),
                # torch.LongTensor(np.asarray(mesh.faces)).to(self.device),
                mesh.faces,
                # torch.FloatTensor(np.asarray(vert_area_weight)).to(self.device),
                # vert_area_weight,
                torch.cat((torch.FloatTensor(vertex_normals), temp), dim=-1).to(self.device).to(torch.float)
                # mesh.vertex_normals,
            ]
        return meshes

    def forward(self, pose, theta):
        """[summary]
        Args:
            pose (Tensor (batch_size x 4 x 4)): The pose of the base link of the hand as a translation matrix.
            theta (Tensor (batch_size x 15)): The seven degrees of freedome of the Barrett hand. The first column specifies the angle between
            fingers F1 and F2,  the second to fourth column specifies the joint angle around the proximal link of each finger while the fifth
            to the last column specifies the joint angle around the distal link for each finger

       """
        batch_size = pose.shape[0]
        pose_normal = pose.clone()

        pose_normal[:, :3, 3] = torch.zeros(3, device=pose.device)


        # rot_z_90 = torch.eye(4, device=self.device)
        #
        # rot_z_90[1, 1] = -1
        # rot_z_90[2, 3] = -0.001 * 79
        # rot_z_90 = rot_z_90.repeat(batch_size, 1, 1)
        # pose = torch.matmul(pose, rot_z_90)
        righthand_vertices = self.righthand.repeat(batch_size, 1, 1)
        righthand_vertices = torch.matmul(torch.matmul(pose, self.T),
                                          righthand_vertices.transpose(2, 1)).transpose(
                                          1, 2)[:, :, :3]

        righthand_vertices_normal = self.righthand_normal.repeat(batch_size, 1, 1)

        righthand_vertices_normal = torch.matmul(torch.matmul(pose_normal, self.T),
                                          righthand_vertices_normal.transpose(2, 1)).transpose(
                                          1, 2)[:, :, :3]
        # palm_1_vertices = self.palm_1.repeat(batch_size, 1, 1)
        # palm_2_vertices = self.palm_2.repeat(batch_size, 1, 1)
        # palm_1_vertices = torch.matmul(pose,
        #                                palm_1_vertices.transpose(2, 1)).transpose(
        #                                  1, 2)[:, :, :3]
        # palm_2_vertices = torch.matmul(pose,
        #                                palm_2_vertices.transpose(2, 1)).transpose(
        #                                  1, 2)[:, :, :3]

        all_base_vertices = torch.zeros(
            (batch_size, 5, self.base.shape[0], 3), device=self.device)      # 5
        all_base_vertices_normal = torch.zeros(
            (batch_size, 5, self.base_normal.shape[0], 3), device=self.device)  # 5
        all_proximal_vertices = torch.zeros(
            (batch_size, 5, self.proximal.shape[0], 3), device=self.device)  # 5
        all_proximal_vertices_normal = torch.zeros(
            (batch_size, 5, self.proximal_normal.shape[0], 3), device=self.device)  # 5
        all_medial_vertices = torch.zeros(
            (batch_size, 5, self.medial.shape[0], 3), device=self.device)  # 5
        all_medial_vertices_normal = torch.zeros(
            (batch_size, 5, self.medial_normal.shape[0], 3), device=self.device)  # 5
        all_distal_vertices = torch.zeros(
            (batch_size, 5, self.distal.shape[0], 3), device=self.device)  # 5
        all_distal_vertices_normal = torch.zeros(
            (batch_size, 5, self.distal_normal.shape[0], 3), device=self.device)  # 5

        base_vertices = self.base.repeat(batch_size, 1, 1)
        base_vertices_normal = self.base_normal.repeat(batch_size, 1, 1)
        proximal_vertices = self.proximal.repeat(batch_size, 1, 1)
        proximal_vertices_normal = self.proximal_normal.repeat(batch_size, 1, 1)
        medial_vertices = self.medial.repeat(batch_size, 1, 1)
        medial_vertices_normal = self.medial_normal.repeat(batch_size, 1, 1)
        distal_vertices = self.distal.repeat(batch_size, 1, 1)
        distal_vertices_normal = self.distal_normal.repeat(batch_size, 1, 1)

        for i in range(5):  # 5
            # print('i is :', i)
            # print(self.A0)
            # print(theta[:, 0+i*4])
            # print(self.phi0)
            # # print(theta[:, 0+i*4]+self.phi0)
            # exit()
            T01 = self.forward_kinematics(self.A0, torch.tensor(0, dtype=torch.float32, device=self.device),
                                          0, -theta[:, 0+i*3]+self.phi0, batch_size)
            T12 = self.forward_kinematics(self.A1, torch.tensor(math.pi/2, dtype=torch.float32, device=self.device),
                                          0, theta[:, 1+i*3]+self.phi1, batch_size)
            # T12 = self.forward_kinematics(self.A1, torch.tensor(0, dtype=torch.float32, device=self.device),
            #                               0, theta[:, 1+i*4]+self.phi1, batch_size)
            T23 = self.forward_kinematics(self.A2, torch.tensor(0, dtype=torch.float32, device=self.device),
                                          0, theta[:, 2+i*3]+self.phi2, batch_size)
            T34 = self.forward_kinematics(self.A3, torch.tensor(0, dtype=torch.float32, device=self.device),
                                          0, theta[:, 2+i*3]+self.phi3, batch_size)

            if i == 0:
                pose_to_Tw0 = torch.matmul(pose, torch.matmul(self.T, self.T_AR))
                pose_to_Tw0_normal = torch.matmul(pose_normal, torch.matmul(self.T, self.T_AR_normal))
            elif i == 1:
                pose_to_Tw0 = torch.matmul(pose,  torch.matmul(self.T, self.T_BR))
                pose_to_Tw0_normal = torch.matmul(pose_normal, torch.matmul(self.T, self.T_BR_normal))
            elif i == 2:
                pose_to_Tw0 = torch.matmul(pose, torch.matmul(self.T, self.T_CR))
                pose_to_Tw0_normal = torch.matmul(pose_normal,  torch.matmul(self.T, self.T_CR_normal))
            elif i == 3:
                pose_to_Tw0 = torch.matmul(pose,  torch.matmul(self.T, self.T_DR))
                pose_to_Tw0_normal = torch.matmul(pose_normal, torch.matmul(self.T, self.T_DR_normal))
            elif i == 4:
                pose_to_Tw0 = torch.matmul(pose,  torch.matmul(self.T, self.T_ER))
                pose_to_Tw0_normal = torch.matmul(pose_normal, torch.matmul(self.T, self.T_ER_normal))

            pose_to_T01 = torch.matmul(pose_to_Tw0, T01)
            pose_to_T01_normal = torch.matmul(pose_to_Tw0_normal, T01)
            # print('matrix shape is :', pose_to_T01.shape)
            # print('shape is :', base_vertices.shape)
            all_base_vertices[:, i] = torch.matmul(
                pose_to_T01,
                base_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

            all_base_vertices_normal[:, i] = torch.matmul(
                pose_to_T01_normal,
                base_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

            pose_to_T12 = torch.matmul(pose_to_T01, T12)
            pose_to_T12_normal = torch.matmul(pose_to_T01_normal, T12)

            all_proximal_vertices[:, i] = torch.matmul(
                pose_to_T12,
                proximal_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

            all_proximal_vertices_normal[:, i] = torch.matmul(
                pose_to_T12_normal,
                proximal_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

            pose_to_T23 = torch.matmul(pose_to_T12, T23)
            pose_to_T23_normal = torch.matmul(pose_to_T12_normal, T23)

            all_medial_vertices[:, i] = torch.matmul(
                pose_to_T23,
                medial_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

            all_medial_vertices_normal[:, i] = torch.matmul(
                pose_to_T23_normal,
                medial_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

            pose_to_T34 = torch.matmul(pose_to_T23, T34)
            pose_to_T34_normal = torch.matmul(pose_to_T23_normal, T34)

            all_distal_vertices[:, i] = torch.matmul(
                pose_to_T34,
                distal_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

            all_distal_vertices_normal[:, i] = torch.matmul(
                pose_to_T34_normal,
                distal_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        return righthand_vertices, all_base_vertices, all_proximal_vertices, all_medial_vertices, all_distal_vertices, \
               righthand_vertices_normal, all_base_vertices_normal, all_proximal_vertices_normal, \
               all_medial_vertices_normal, all_distal_vertices_normal

    def forward_kinematics(self, A, alpha, D, theta, batch_size=1):
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        c_alpha = torch.cos(alpha)
        s_alpha = torch.sin(alpha)
        l_1_to_l = torch.zeros((batch_size, 4, 4), device=self.device)
        l_1_to_l[:, 0, 0] = c_theta
        l_1_to_l[:, 0, 1] = -s_theta
        l_1_to_l[:, 0, 3] = A
        l_1_to_l[:, 1, 0] = s_theta * c_alpha
        l_1_to_l[:, 1, 1] = c_theta * c_alpha
        l_1_to_l[:, 1, 2] = -s_alpha
        l_1_to_l[:, 1, 3] = -s_alpha * D
        l_1_to_l[:, 2, 0] = s_theta * s_alpha
        l_1_to_l[:, 2, 1] = c_theta * s_alpha
        l_1_to_l[:, 2, 2] = c_alpha
        l_1_to_l[:, 2, 3] = c_alpha * D
        l_1_to_l[:, 3, 3] = 1
        return l_1_to_l

    def get_hand_mesh(self, vertices_list, faces, save_mesh=True, path='./output_mesh'):
        if save_mesh:
            assert os.path.exists(path)

        righthand_verts = vertices_list[0]
        righthand_faces = faces[0]

        # palm_2_verts = vertices_list[1]
        # palm_2_faces = faces[1]
        # save_to_mesh(palm_2_verts, palm_2_faces, '{}/hitdlr_palm2.obj'.format(path))

        thumb_base_verts = vertices_list[1][0]
        thumb_base_faces = faces[1]

        thumb_proximal_verts = vertices_list[2][0]
        thumb_proximal_faces = faces[2]

        thumb_medial_verts = vertices_list[3][0]
        thumb_medial_faces = faces[3]

        thumb_distal_verts = vertices_list[4][0]
        thumb_distal_faces = faces[4]

        fore_base_verts = vertices_list[1][1]
        fore_base_faces = faces[1]

        fore_proximal_verts = vertices_list[2][1]
        fore_proximal_faces = faces[2]

        fore_medial_verts = vertices_list[3][1]
        fore_medial_faces = faces[3]

        fore_distal_verts = vertices_list[4][1]
        fore_distal_faces = faces[4]

        middle_base_verts = vertices_list[1][2]
        middle_base_faces = faces[1]

        middle_proximal_verts = vertices_list[2][2]
        middle_proximal_faces = faces[2]

        middle_medial_verts = vertices_list[3][2]
        middle_medial_faces = faces[3]

        middle_distal_verts = vertices_list[4][2]
        middle_distal_faces = faces[4]

        ring_base_verts = vertices_list[1][3]
        ring_base_faces = faces[1]

        ring_proximal_verts = vertices_list[2][3]
        ring_proximal_faces = faces[2]

        ring_medial_verts = vertices_list[3][3]
        ring_medial_faces = faces[3]

        ring_distal_verts = vertices_list[4][3]
        ring_distal_faces = faces[4]

        little_base_verts = vertices_list[1][4]
        little_base_faces = faces[1]

        little_proximal_verts = vertices_list[2][4]
        little_proximal_faces = faces[2]

        little_medial_verts = vertices_list[3][4]
        little_medial_faces = faces[3]

        little_distal_verts = vertices_list[4][4]
        little_distal_faces = faces[4]

        if save_mesh:
            save_to_mesh(righthand_verts, righthand_faces, '{}/hitdlr_righthand.obj'.format(path))
            save_to_mesh(thumb_base_verts, thumb_base_faces, '{}/hitdlr_thumb_base.obj'.format(path))
            save_to_mesh(thumb_proximal_verts, thumb_proximal_faces, '{}/hitdlr_thumb_proximal.obj'.format(path))
            save_to_mesh(thumb_medial_verts, thumb_medial_faces, '{}/hitdlr_thumb_medial.obj'.format(path))
            save_to_mesh(thumb_distal_verts, thumb_distal_faces, '{}/hitdlr_thumb_distal.obj'.format(path))
            save_to_mesh(fore_base_verts, fore_base_faces, '{}/hitdlr_fore_base.obj'.format(path))
            save_to_mesh(fore_proximal_verts, fore_proximal_faces, '{}/hitdlr_fore_proximal.obj'.format(path))
            save_to_mesh(fore_medial_verts, fore_medial_faces, '{}/hitdlr_fore_medial.obj'.format(path))
            save_to_mesh(fore_distal_verts, fore_distal_faces, '{}/hitdlr_fore_distal.obj'.format(path))
            save_to_mesh(middle_base_verts, middle_base_faces, '{}/hitdlr_middle_base.obj'.format(path))
            save_to_mesh(middle_proximal_verts, middle_proximal_faces, '{}/hitdlr_middle_proximal.obj'.format(path))
            save_to_mesh(middle_medial_verts, middle_medial_faces, '{}/hitdlr_middle_medial.obj'.format(path))
            save_to_mesh(middle_distal_verts, middle_distal_faces, '{}/hitdlr_middle_distal.obj'.format(path))
            save_to_mesh(ring_base_verts, ring_base_faces, '{}/hitdlr_ring_base.obj'.format(path))
            save_to_mesh(ring_proximal_verts, ring_proximal_faces, '{}/hitdlr_ring_proximal.obj'.format(path))
            save_to_mesh(ring_medial_verts, ring_medial_faces, '{}/hitdlr_ring_medial.obj'.format(path))
            save_to_mesh(ring_distal_verts, ring_distal_faces, '{}/hitdlr_ring_distal.obj'.format(path))
            save_to_mesh(little_base_verts, little_base_faces, '{}/hitdlr_little_base.obj'.format(path))
            save_to_mesh(little_proximal_verts, little_proximal_faces, '{}/hitdlr_little_proximal.obj'.format(path))
            save_to_mesh(little_medial_verts, little_medial_faces, '{}/hitdlr_little_medial.obj'.format(path))
            save_to_mesh(little_distal_verts, little_distal_faces, '{}/hitdlr_little_distal.obj'.format(path))

            hand_mesh = []
            for root, dirs, files in os.walk('{}'.format(path)):
                for filename in files:
                    if filename.endswith('.obj'):
                        filepath = os.path.join(root, filename)
                        mesh = trimesh.load_mesh(filepath)
                        hand_mesh.append(mesh)
            hand_mesh = np.sum(hand_mesh)
        else:
            righthand_mesh = trimesh.Trimesh(righthand_verts, righthand_faces)
            # righthand_mesh.visual.face_colors = [128,128,128,255]

            thumb_proximal_mesh = trimesh.Trimesh(thumb_proximal_verts, thumb_proximal_faces)
            # thumb_proximal_mesh.visual.face_colors = [255,128,0,255]
            # thumb_proximal_mesh.show()
            thumb_medial_mesh = trimesh.Trimesh(thumb_medial_verts, thumb_medial_faces)
            # thumb_medial_mesh.visual.face_colors = [255,0,60,255]
            # thumb_medial_mesh.show()
            thumb_distal_mesh = trimesh.Trimesh(thumb_distal_verts, thumb_distal_faces)
            # thumb_distal_mesh.visual.face_colors = [255,0,60,255]
            # thumb_distal_mesh.show()

            fore_proximal_mesh = trimesh.Trimesh(fore_proximal_verts, fore_proximal_faces)
            # fore_proximal_mesh.visual.face_colors = [255,128,0,255]
            fore_medial_mesh = trimesh.Trimesh(fore_medial_verts, fore_medial_faces)
            # fore_medial_mesh.visual.face_colors = [0,255,128,255]
            fore_distal_mesh = trimesh.Trimesh(fore_distal_verts, fore_distal_faces)
            # fore_distal_mesh.visual.face_colors = [0,128,255,255]
            #

            middle_proximal_mesh = trimesh.Trimesh(middle_proximal_verts, middle_proximal_faces)
            # middle_proximal_mesh.visual.face_colors = [255,128,0,255]
            middle_medial_mesh = trimesh.Trimesh(middle_medial_verts, middle_medial_faces)
            # middle_medial_mesh.visual.face_colors = [0,255,128,255]
            middle_distal_mesh = trimesh.Trimesh(middle_distal_verts, middle_distal_faces)
            # middle_distal_mesh.visual.face_colors = [0,128,255,255]

            ring_proximal_mesh = trimesh.Trimesh(ring_proximal_verts, ring_proximal_faces)
            # ring_proximal_mesh.visual.face_colors = [255,128,0,255]
            ring_medial_mesh = trimesh.Trimesh(ring_medial_verts, ring_medial_faces)
            # ring_medial_mesh.visual.face_colors = [0,255,128,255]
            ring_distal_mesh = trimesh.Trimesh(ring_distal_verts, ring_distal_faces)
            # ring_distal_mesh.visual.face_colors = [0,128,255,255]

            little_proximal_mesh = trimesh.Trimesh(little_proximal_verts, little_proximal_faces)
            # little_proximal_mesh.visual.face_colors = [255,128,0,255]
            little_medial_mesh = trimesh.Trimesh(little_medial_verts, little_medial_faces)
            # little_medial_mesh.visual.face_colors = [0,255,128,255]
            little_distal_mesh = trimesh.Trimesh(little_distal_verts, little_distal_faces)
            # little_distal_mesh.visual.face_colors = [0,128,255,255]
            # hand_mesh = [righthand_mesh,
            #              thumb_base_mesh, thumb_proximal_mesh, thumb_medial_mesh, thumb_distal_mesh,
            #              fore_base_mesh, fore_proximal_mesh, fore_medial_mesh, fore_distal_mesh,
            #              middle_base_mesh, middle_proximal_mesh, middle_medial_mesh, middle_distal_mesh,
            #              ring_base_mesh, ring_proximal_mesh, ring_medial_mesh, ring_distal_mesh,
            #              little_base_mesh, little_proximal_mesh, little_medial_mesh, little_distal_mesh
            #              ]
            hand_mesh = [righthand_mesh,
                         thumb_proximal_mesh, thumb_medial_mesh, thumb_distal_mesh,
                         fore_proximal_mesh, fore_medial_mesh, fore_distal_mesh,
                         middle_proximal_mesh, middle_medial_mesh, middle_distal_mesh,
                         ring_proximal_mesh, ring_medial_mesh, ring_distal_mesh,
                         little_proximal_mesh, little_medial_mesh, little_distal_mesh
                         ]
            # np.sum(hand_mesh).show()
        return hand_mesh

    def get_forward_hand_mesh(self, pose, theta, save_mesh=False, path='./output_mesh'):
        batch_size = pose.size()[0]
        outputs = self.forward(pose, theta)[:5]

        # hand_vertices_list = [output.squeeze().detach().cpu().numpy() for output in outputs]
        hand_vertices_list = [[outputs[0][i].detach().cpu().numpy(), outputs[1][i].detach().cpu().numpy(),
                               outputs[2][i].detach().cpu().numpy(), outputs[3][i].detach().cpu().numpy(),
                               outputs[4][i].detach().cpu().numpy()] for i in range(batch_size)]

        hand_meshes = [self.get_hand_mesh(hand_vertices, self.gripper_faces, save_mesh=save_mesh, path=path) for hand_vertices in hand_vertices_list]
        hand_meshes = [np.sum(hand_mesh) for hand_mesh in hand_meshes]
        return hand_meshes

    def get_forward_vertices(self, pose, theta):
        batch_size = pose.size()[0]
        outputs = self.forward(pose, theta)

        hand_vertices = torch.cat((outputs[0].view(batch_size, -1, 3),
                                   # outputs[1].view(batch_size, -1, 3),
                                   outputs[2].view(batch_size, -1, 3),
                                   outputs[3].view(batch_size, -1, 3),
                                   outputs[4].view(batch_size, -1, 3)), 1)  # .squeeze()

        hand_vertices_normal = torch.cat((outputs[5].view(batch_size, -1, 3),
                                          # outputs[6].view(batch_size, -1, 3),
                                          outputs[7].view(batch_size, -1, 3),
                                          outputs[8].view(batch_size, -1, 3),
                                          outputs[9].view(batch_size, -1, 3)), 1)  # .squeeze()

        return hand_vertices, hand_vertices_normal

    def get_faces(self):
        all_faces = np.concatenate((np.asarray(self.meshes['righthand_base'][1]),
                                    np.asarray(self.meshes['proximal'][1])+648,
                                    np.asarray(self.meshes['proximal'][1])+648+422,
                                    np.asarray(self.meshes['proximal'][1])+648+422*2,
                                    np.asarray(self.meshes['proximal'][1])+648+422*3,
                                    np.asarray(self.meshes['proximal'][1])+648+422*4,
                                    np.asarray(self.meshes['medial'][1])+648+422*5,
                                    np.asarray(self.meshes['medial'][1])+648+422*5+314,
                                    np.asarray(self.meshes['medial'][1])+648+422*5+314*2,
                                    np.asarray(self.meshes['medial'][1])+648+422*5+314*3,
                                    np.asarray(self.meshes['medial'][1])+648+422*5+314*4,
                                    np.asarray(self.meshes['distal'][1])+648+422*5+314*5,
                                    np.asarray(self.meshes['distal'][1])+648+422*5+314*5+402,
                                    np.asarray(self.meshes['distal'][1])+648+422*5+314*5+402*2,
                                    np.asarray(self.meshes['distal'][1])+648+422*5+314*5+402*3,
                                    np.asarray(self.meshes['distal'][1])+648+422*5+314*5+402*4,
                                    ), axis=0)
        return all_faces


if __name__ == "__main__":
    device = 'cuda'
    taxonomy = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch','Precision_Sphere','Large_Wrap']
    hit = HitdlrLayer(device).to(device)
    # print(hit.gripper_faces)
    # exit()
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    # pose = [[[0, 1, 0, 0.1],
    #          [-1, 0, 0, 0.1],
    #          [0, 0, 1, 0.1],
    #          [0, 0, 0, 1]]]
    # pose = torch.tensor(pose, dtype=torch.float32, device=device, requires_grad=False)
    # pose = pose.repeat(2, 1, 1)

    # theta = np.zeros((1, 15), dtype=np.float32)
    tax = taxonomy[-2]
    theta = np.array(0.5*np.array(grasp_dict_20f[tax]["joint_init"]) +
                     0.5*np.array(grasp_dict_20f[tax]["joint_final"]))*np.pi/180.
    theta = np.array([theta[0], theta[1], theta[2],
                      theta[4], theta[5], theta[6],
                      theta[8], theta[9], theta[10],
                      theta[12], theta[13], theta[14],
                      theta[16], theta[17], theta[18]])[np.newaxis,:]
    theta = torch.from_numpy(theta).to(device)

    scene = trimesh.Scene()
    meshes = hit.get_forward_hand_mesh(pose, theta, save_mesh=False, path='./output_mesh')[0]
    scene.add_geometry(meshes)
    # scene.show()
    # exit()
    hand_verts, hand_normal = hit.get_forward_vertices(pose, theta)
    hand_verts = np.asarray(hand_verts.cpu())[0]
    print(hand_verts.shape)
    hand_normal = np.asarray(hand_normal.cpu())[0]

    all_points = TaxPoints[tax]
    color = np.zeros([hand_verts.shape[0], 3])
    color[all_points] = [255, 0, 0]
    print(len(all_points))
    v = trimesh.PointCloud(hand_verts[all_points], [255, 0, 0])
    scene.add_geometry([v])
    scene.show()
    # exit()
    # v.show()
    # exit()

    ray_visualize = trimesh.load_path(np.hstack((hand_verts, hand_verts + hand_normal / 100)).reshape(-1, 2, 3))
    scene = trimesh.Scene()
    scene.add_geometry([meshes,v])
    scene.show()
    exit()

    # meshes.show()
    # for key in grasp_dict_15f.keys():
    #     print(key)
    #     theta = (np.radians(np.array(grasp_dict_15f[key]['joint_init']).astype(np.float32)) + \
    #             2 * np.radians(np.array(grasp_dict_15f[key]['joint_final']).astype(np.float32))) / 3
    #     theta = torch.from_numpy(theta).to(device).reshape(-1, 15)
    #     meshes = hit.get_forward_hand_mesh(pose, theta, save_mesh=False, path='./output_mesh')[0]
    #     meshes.show()
    exit()
    # theta = theta.repeat(2, 1)
    # print(theta.shape)

    # theta = torch.zeros((1, 20), dtype=torch.float32).to(device)

    # theta[0, 4] = 1
    # theta[0, 1] = 0.5
    # theta[0, 2] = 0.0
    # theta[0, 3] = 0.0
    import time
    timestart = time.time()
    meshes = hit.get_forward_hand_mesh(pose, theta, save_mesh=False, path='./output_mesh')
    # meshes[0].visual.face_colors = [73*1.5, 86*1.5, 87*1.5]
    meshes[0].visual.face_colors = [180, 176, 177]
    # print(meshes[0].vertices.shape)
    # print(meshes[0].faces.shape)
    # np.save('hand_mesh_face.npy', meshes[0].faces)

    points = np.array([
                       [3., 120, -10], [3., 125, -10],
                       [3., 130, -10], [3., 135, -10],
                       [3., 140, -10], [3., 145, -10],
                       [3., 150, -10], [3., 155, -10],
                       [3., 160, -10], [3., 165, -10],
                       [3., 170, -10], [3., 175, -10],
                       [3., 180, -10], [3., 185, -10],
                       [3., 190, -10], [3., 195, -10],
                       [3., 200, -10], [3., 205, -10],
                       [3., 210, -10], [3., 215, -10],


                       # [27.3651, 74.2506, 7.24275],
                       # [-20.8083, 76.9719, 8.81745],
                       # [-45.5689, 58.4249, 14.4583],
                       # [21.2862, 63.0753, 23.4334],
                       # [-13.3113, 15.0035, 0.0253414],
                       # [-28.6422, 15.6759, -0.0355567],  [-11.4784, 14.2419, -0.0250446], [26.5796, 2.5569, -0.0611752],
                       # [21.4439, 8.61383, 0.0853162], [11.6691, 10.9797, 0.137127]
                       ]).reshape(-1, 3) / 1000.0
    # points = points[[1, 3, 5, 7, 9, 11, 13, 15, 17]]
    points = points[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]]
    points = points - np.array([0, 1.75, 0]) / 1000.0

    # points = np.array([[1.96607, 100.8198, 0.84926], [27.3651, 92.2506, 2.24275], [-20.8083, 94.9719, 3.81745],
    #                    [-45.5689, 76.4249, 9.4583], [21.2862, 81.0753, 28.4334]]) / 1000.0

    promixity_query = trimesh.proximity.ProximityQuery(meshes[0])
    _, closest_vertex_idxs_on_mesh = promixity_query.vertex(points)
    print(closest_vertex_idxs_on_mesh)

    closest_vertices_on_mesh = meshes[0].vertices[closest_vertex_idxs_on_mesh]

    pc_1 = trimesh.PointCloud(points.reshape(-1, 3), colors=[255, 0, 0])
    pc_2 = trimesh.PointCloud(closest_vertices_on_mesh.reshape(-1, 3), colors=[255, 255, 0])
    scene = trimesh.Scene([meshes[0], pc_1, pc_2])
    scene.show()

    # exit()
    # point_a = np.random.randn(1, 3) * 0.1
    # point_a[0] = np.array([0, 0, 0])
    #
    # pc = trimesh.PointCloud(point_a, colors=[0, 255, 0])
    #
    # distance_min = - np.inf
    # for mesh in meshes[0].split():
    #     distance = trimesh.proximity.signed_distance(mesh, point_a)
    #     print(distance)
    #     if distance > distance_min:
    #         distance_min = distance
    #     print(distance)
    # print(distance_min)

    show_vertices = True
    if show_vertices:
        vertices, normal = hit.get_forward_vertices(pose, theta)
        from utils_.farthest_points_sampling import fps
        ray_origins = vertices.squeeze().detach().cpu().numpy()
        ray_directions = normal.squeeze().detach().cpu().numpy()
        idxs = fps(ray_origins, 2048)
        # np.save('./picked_hand_points.npy', idxs)

        sampled_pc = ray_origins[idxs]
        pc = trimesh.PointCloud(ray_origins, colors=[0, 255, 0])

        ray_visualize = trimesh.load_path(np.hstack((ray_origins, ray_origins + ray_directions / 100)).reshape(-1, 2, 3))
        # scene = trimesh.Scene([pc, ray_visualize])
        # idxs = [790, 1212, 1634, 2056, 2478, 2991, 3305, 3619, 3933, 4247, 4443, 4450, 4845, 5247, 5649, 6051]
        # closest_vertex_idxs_on_mesh = [536, 141, 38, 0, 344, 790, 1212, 1634, 2056, 2478, 2991, 3305, 3619, 3933, 4247, 4443, 4450, 4845,
        #                                4852, 5247, 5254, 5649, 5656, 6051, 6058]

        # closest_vertex_idxs_on_mesh = [3573-314*2, 3619-314*2, 3553-314*2, 3573-314*1, 3619-314*1, 3553-314*1, 3573-314*0, 3619-314*0, 3553-314*0, 3573-314*-1, 3619-314*-1, 3553-314*-1, 3573-314*-2, 3619-314*-2, 3553-314*-2]
        closest_vertex_idxs_on_mesh = [894, 931, 1023, 1316, 1353, 1445, 1738, 1775, 1867, 2160, 2197, 2289, 2582, 2619, 2711,
                                       2945, 2991, 2925, 3259, 3305, 3239, 3573, 3619, 3553, 3887, 3933, 3867, 4201, 4247, 4181,
                                       4491, 4605, 4381, 4443, 4893, 5007, 4783, 4845, 5295, 5409, 5185, 5247, 5697, 5811, 5587, 5649, 6099, 6213, 5989, 6051]
        # [3573, 3619, 3553, 5295, 5409, 5185, 5181]
        # closest_vertex_idxs_on_mesh = [2991, 3305, 3619, 3933, 4247, 4443, 4450, 4845,
        #                                4852, 5247, 5254, 5649, 5656, 6051, 6058]
        # np.save('contact_idxs.npy', np.asarray(closest_vertex_idxs_on_mesh).astype(int))

        # import scipy
        # point_tree = scipy.spatial.KDTree(ray_origins)
        # distances, idxs = point_tree.query(points, k=1)
        # print(idxs)
        grasp_points = ray_origins[closest_vertex_idxs_on_mesh]
        print(grasp_points.shape)

        pc_3 = trimesh.PointCloud(ray_origins[closest_vertex_idxs_on_mesh].reshape(-1, 3), colors=[255, 0, 0])
        # scene = trimesh.Scene([pc, pc_3])
        scene = trimesh.Scene()
        meshes[0].visual.face_colors[:] = [158, 158, 158, 250]
        scene.add_geometry(meshes[0])
        for i in range(grasp_points.shape[0]):
            ball = trimesh.creation.icosphere(subdivisions=3, radius=0.0025, color=[255, 0, 0])
            ball.apply_translation(grasp_points[i])
            scene.add_geometry(ball)
        scene.show()
    # super_mesh = np.sum(mesh)
    print('time cost is: ', time.time() - timestart)

    # print(super_mesh.vertices.shape)
    # exit()
    # super_mesh.show()
    # T = torch.from_numpy(np.load('./T.npy').astype(np.float32))
    # mesh.apply_transform(T)
    # mesh.show()
    # mesh.apply_transofrmation(T)

    # mesh.show()
    # super_mesh = trimesh.load_mesh('~/super_mesh.stl')
    # super_mesh.visual.face_colors = [0, 255, 0]
    # scene = trimesh.Scene([mesh, super_mesh])
    # scene.show()
    # (mesh + super_mesh).show()

    # outputs = hit.forward(pose, theta=theta)
    # vertices_list = [output.squeeze().detach().cpu().numpy() for output in outputs]
    # print(vertices[2].shape)
    # print(vertices[3].shape)
    # save_hand(vertices_list, hit.gripper_faces)
    # palm_1_pc = trimesh.PointCloud(vertices[0], colors=[0, 255, 0])
    # palm_2_pc = trimesh.PointCloud(vertices[1], colors=[0, 255, 0])
    #
    # save_to_mesh(vertices, hit.gripper_faces)
    # mesh = trimesh.load_mesh('./hitdlr.obj')
    # mesh.show()
    # exit()
    # base_pc = trimesh.PointCloud(vertices[2].reshape(-1, 3), colors=[0, 255, 0])
    # proximal_pc = trimesh.PointCloud(vertices[3].reshape(-1, 3), colors=[0, 255, 0])
    # medial_pc = trimesh.PointCloud(vertices[4].reshape(-1, 3), colors=[0, 255, 0])
    # distal_pc = trimesh.PointCloud(vertices[5].reshape(-1, 3), colors=[0, 255, 0])
    # # print(vertices[5])
    # # print(distal_pc)
    # scene = trimesh.Scene([palm_1_pc, palm_2_pc, base_pc, proximal_pc, medial_pc, distal_pc])
    # # scene = trimesh.Scene([base_pc, proximal_pc, medial_pc, distal_pc])
    # scene.show()






