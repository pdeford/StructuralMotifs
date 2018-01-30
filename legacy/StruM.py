#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr
import StringIO
import types


diprodb = StringIO.StringIO("""ID	PropertyName	AA	AC	AG	AT	CA	CC	CG	CT	GA	GC	GG	GT	TA	TC	TG	TT	NucleicAcid	Strand	
1	Twist	38.9	31.12	32.15	33.81	41.41	34.96	32.91	32.15	41.31	38.5	34.96	31.12	33.28	41.31	41.41	38.9	B-DNA	double	
2	Stacking energy	-12	-11.8	-11.5	-10.6	-12.3	-9.5	-13.1	-11.5	-11.4	-13.2	-9.5	-11.8	-11.2	-11.4	-12.3	-12	B-DNA	double	
3	Rise	3.16	3.41	3.63	3.89	3.23	4.08	3.6	3.63	3.47	3.81	4.08	3.41	3.21	3.47	3.23	3.16	B-DNA	double	
4	Bend	3.07	2.97	2.31	2.6	3.58	2.16	2.81	2.31	2.51	3.06	2.16	2.97	6.74	2.51	3.58	3.07	B-DNA	double	
5	Tip	1.76	2	0.9	1.87	-1.64	0.71	0.22	0.9	1.35	2.5	0.71	2	6.7	1.35	-1.64	1.76	B-DNA	double	
6	Inclination	-1.43	-0.11	-0.92	0	1.31	-1.11	0	0.92	-0.33	0	1.11	0.11	0	0.33	-1.31	1.43	DNA	single	
7	Major Groove Width	12.15	12.37	13.51	12.87	13.58	15.49	14.42	13.51	13.93	14.55	15.49	12.37	12.32	13.93	13.58	12.15	B-DNA	double	
8	Major Groove Depth	9.12	9.41	8.96	8.96	8.67	8.45	8.81	8.96	8.76	8.67	8.45	9.41	9.6	8.76	8.67	9.12	B-DNA	double	
9	Major Groove Size	3.98	3.98	4.7	4.7	3.98	3.98	4.7	4.7	3.26	3.26	3.98	3.98	3.26	3.26	3.98	3.98	B-DNA	double	
10	Major Groove Distance	3.38	3.03	3.36	3.02	3.79	3.38	3.77	3.36	3.4	3.04	3.38	3.03	3.81	3.4	3.79	3.38	B-DNA	double	
11	Minor Groove Width	5.3	6.04	5.19	5.31	4.79	4.62	5.16	5.19	4.71	4.74	4.62	6.04	6.4	4.71	4.79	5.3	B-DNA	double	
12	Minor Groove Depth	9.03	8.79	8.98	8.91	9.09	8.99	9.06	8.98	9.11	8.98	8.99	8.79	9	9.11	9.09	9.03	B-DNA	double	
13	Minor Groove Size	2.98	3.26	3.98	3.26	3.7	3.98	4.7	3.98	2.98	3.26	3.98	3.26	2.7	2.98	3.7	2.98	B-DNA	double	
14	Minor Groove Distance	2.94	4.22	2.79	4.2	3.09	2.8	3.21	2.79	2.95	4.24	2.8	4.22	2.97	2.95	3.09	2.94	B-DNA	double	
15	Persistance Length	35	60	60	20	60	130	85	60	60	85	130	60	20	60	60	35	B-DNA	double	
16	Melting Temperature	54.5	97.73	58.42	57.02	54.71	85.97	72.55	58.42	86.44	136.12	85.97	97.73	36.73	86.44	54.71	54.5	B-DNA	double	
17	Probability contacting nucleosome core	18.4	10.2	14.5	7.2	15.7	10.2	1.1	14.5	11.3	5.2	10.2	10.2	6.2	11.3	15.7	18.4	B-DNA	double	
18	Mobility to bend towards major groove	1.18	1.06	1.06	1.12	1.06	0.99	1.02	1.04	1.08	0.98	1	1.02	1.07	1.03	1.03	1.09	DNA	single	
19	Mobility to bend towards minor groove	1.04	1.1	1.09	1.02	1.16	1.27	1.25	1.16	1.12	1.17	1.25	1.11	1.05	1.2	1.23	1.04	DNA	single	
20	Propeller Twist	-17.3	-6.7	-14.3	-16.9	-8.6	-12.8	-11.2	-14.3	-15.1	-11.7	-12.8	-6.7	-11.1	-15.1	-8.6	-17.3	B-DNA	double	
21	Clash Strength	0.64	0.95	2.53	1.68	0.8	1.78	2.42	2.53	0.03	0.22	1.78	0.95	0	0.03	0.8	0.64	B-DNA	double	
22	Enthalpy	-8	-9.4	-6.6	-5.6	-8.2	-10.9	-11.8	-6.6	-8.8	-10.5	-10.9	-9.4	-6.6	-8.8	-8.2	-8	B-DNA	double	
23	Entropy	-21.9	-25.5	-16.4	-15.2	-21	-28.4	-29	-16.4	-23.5	-26.4	-28.4	-25.5	-18.4	-23.5	-21	-21.9	B-DNA	double	
24	Shift (RNA)	-0.08	0.23	-0.04	-0.06	0.11	-0.01	0.3	-0.04	0.07	0.07	-0.01	0.23	-0.02	0.07	0.11	-0.08	A-RNA	double	
25	Roll (DNA-protein complex)	0.8	-0.2	5.6	0	6.4	3.3	6.5	5.6	2.4	-2	3.3	-0.2	2.7	2.4	6.4	0.8	B-DNA	double	
26	Twist (DNA-protein complex)	35.6	31.1	31.9	29.3	35.9	33.3	34.9	31.9	35.9	34.6	33.3	31.1	39.5	35.9	36	35.6	B-DNA	double	
27	Tilt (DNA-protein complex)	1.9	0.3	1.3	0	0.3	1	0	1.3	1.7	0	1	-0.1	0	1.7	0.3	1.9	B-DNA	double	
28	Slide (DNA-protein complex)	0.1	-0.6	-0.3	-0.7	0.4	-0.1	0.7	-0.3	0.1	-0.3	-0.1	-0.6	0.1	0.1	0.4	0.1	B-DNA	double	
29	Hydrophilicity (RNA)	0.023	0.083	0.035	0.09	0.118	0.349	0.193	0.378	0.048	0.146	0.065	0.16	0.112	0.359	0.224	0.389	RNA	single	
30	Shift (DNA-protein complex)	0.1	-0.1	-0.2	0	0	0	0	-0.2	0.3	0	0	-0.1	0	0.3	0	0.1	B-DNA	double	
31	Hydrophilicity (RNA)	0.04	0.14	0.08	0.14	0.21	0.49	0.35	0.52	0.1	0.26	0.17	0.27	0.21	0.48	0.34	0.44	RNA	single	
32	Rise (DNA-protein complex)	3.3	3.4	3.4	3.3	3.4	3.4	3.4	3.4	3.4	3.4	3.4	3.4	3.4	3.4	3.4	3.3	B-DNA	double	
33	Stacking energy	-5.37	-10.51	-6.78	-6.57	-6.57	-8.26	-9.69	-6.78	-9.81	-14.59	-8.26	-10.51	-3.82	-9.81	-6.57	-5.37	B-DNA	double	
34	Free energy	-0.67	-1.28	-1.17	-0.62	-1.19	-1.55	-1.87	-1.17	-1.12	-1.85	-1.55	-1.28	-0.7	-1.12	-1.19	-0.67	B-DNA	double	
35	Free energy	-1.66	-1.13	-1.35	-1.19	-1.8	-2.75	-3.28	-1.35	-1.41	-2.82	-2.75	-1.13	-0.76	-1.41	-1.8	-1.66	B-DNA	double	
36	Free energy	-0.89	-1.35	-1.16	-0.81	-1.37	-1.64	-1.99	-1.16	-1.25	-1.96	-1.64	-1.35	-0.81	-1.16	-1.37	-0.89	B-DNA	double	
37	Twist (DNA-protein complex)	35.1	31.5	31.9	29.3	37.3	32.9	36.1	31.9	36.3	33.6	32.9	31.5	37.8	36.3	37.3	35.1	B-DNA	double	
38	Free energy	-0.43	-0.98	-0.83	-0.27	-0.97	-1.22	-1.7	-0.83	-0.93	-1.64	-1.22	-0.98	-0.22	-0.93	-0.97	-0.43	B-DNA	double	
39	Twist_twist	0.0461	0.0489	0.0441	0.0463	0.021	0.0482	0.0227	0.0441	0.0422	0.0421	0.0482	0.0489	0.0357	0.0422	0.021	0.0461	B-DNA	double	
40	Tilt_tilt	0.0389	0.0411	0.0371	0.0404	0.0275	0.0414	0.0278	0.0371	0.0392	0.0396	0.0414	0.0411	0.0245	0.0392	0.0275	0.0389	B-DNA	double	
41	Roll_roll	0.0235	0.0267	0.0227	0.0272	0.0184	0.0241	0.0153	0.0227	0.0211	0.0275	0.0241	0.0267	0.0136	0.0211	0.0184	0.0235	B-DNA	double	
42	Twist_tilt	0.006	0.0007	-0.0027	-0.0003	-0.0005	-0.0004	0.0014	-0.0027	0.0005	0.0002	-0.0004	0.0007	-0.0008	0.0005	-0.0005	0.006	B-DNA	double	
43	Twist_roll	0.0083	0.0076	0.0057	0.0081	0.0049	0.0044	0.0031	0.0057	0.0086	0.007	0.0044	0.0076	0.0084	0.0086	0.0049	0.0083	B-DNA	double	
44	Tilt_roll	0.0033	0.0029	-0.0027	0.0007	0.0009	-0.0009	0.0011	-0.0027	-0.0002	-0.001	-0.0009	0.0029	-0.0001	-0.0002	0.0009	0.0033	B-DNA	double	
45	Shift_shift	1.9748	1.341	1.6568	1.1932	1.6003	1.9839	1.3464	1.6568	1.4302	1.7614	1.9839	1.341	1.5294	1.4302	1.6003	1.9748	B-DNA	double	
46	Slide_slide	2.9137	2.9739	2.7056	3.3095	2.2856	3.2154	2.0342	2.7056	2.5179	2.7084	3.2154	2.9739	2.2691	2.5179	2.2856	2.9137	B-DNA	double	
47	Rise_rise	7.6206	9.8821	6.3875	10.4992	6.2903	7.3347	4.3896	6.3875	8.3295	10.2808	7.3347	9.8821	5.0546	8.3295	6.2903	7.6206	B-DNA	double	
48	Shift_slide	0.1711	-0.1574	-0.0263	-0.0965	-0.2832	0.0572	-0.1867	-0.0263	0.0259	0.3178	0.0572	-0.1574	0.0516	0.0259	-0.2832	0.1711	B-DNA	double	
49	Shift_rise	0.1922	-0.0059	-0.0318	-0.0231	-0.0651	0.2151	-0.0411	-0.0318	0.025	0.1312	0.2151	-0.0059	-0.033	0.025	-0.0651	0.1922	B-DNA	double	
50	Slide_rise	1.3815	2.5929	1.3204	2.4811	0.816	1.1959	1.4671	1.3204	1.1528	2.5578	1.1959	2.5929	0.913	1.1528	0.816	1.3815	B-DNA	double	
51	Twist_shift	0.0568	0.0051	-0.0311	-0.0082	-0.0102	0.0238	0.0226	-0.0311	-0.0011	-0.0012	0.0238	0.0051	-0.0058	-0.0011	-0.0102	0.0568	B-DNA	double	
52	Twist_slide	-0.218	-0.2007	-0.1764	-0.1157	-0.017	-0.225	-0.0855	-0.1764	-0.2056	-0.1929	-0.225	-0.2007	-0.0926	-0.2056	-0.017	-0.218	B-DNA	double	
53	Twist_rise	-0.1587	-0.16	-0.1437	-0.0891	-0.1259	-0.1142	-0.1243	-0.1437	-0.1276	-0.1603	-0.1142	-0.16	-0.0932	-0.1276	-0.1259	-0.1587	B-DNA	double	
54	Tilt_shift	0.0015	-0.0049	-0.0194	0.0241	0.004	-0.0653	-0.0516	-0.0194	-0.0262	-0.0478	-0.0653	-0.0049	0.0233	-0.0262	0.004	0.0015	B-DNA	double	
55	Tilt_slide	-0.0075	-0.0129	0.0078	-0.0097	-0.0021	0.005	0.0103	0.0078	-0.0023	-0.0183	0.005	-0.0129	0.0052	-0.0023	-0.0021	-0.0075	B-DNA	double	
56	Tilt_rise	-0.2054	0.0439	0.0498	0.0063	-0.0158	-0.0838	0.0047	0.0498	-0.0829	-0.0632	-0.0838	0.0439	-0.0032	-0.0829	-0.0158	-0.2054	B-DNA	double	
57	Roll_shift	0.0158	0.0141	-0.0143	0.009	-0.0024	-0.0042	0.0106	-0.0143	0.0112	-0.0015	-0.0042	0.0141	-0.0097	0.0112	-0.0024	0.0158	B-DNA	double	
58	Roll_slide	-0.022	-0.0022	-0.0291	-0.0499	0.0093	-0.007	-0.0205	-0.0291	-0.0006	0.0055	-0.007	-0.0022	-0.0078	-0.0006	0.0093	-0.022	B-DNA	double	
59	Roll_rise	-0.0541	0.1089	-0.001	0.0927	-0.0865	0.0044	-0.0199	-0.001	-0.0121	0.1257	0.0044	0.1089	-0.037	-0.0121	-0.0865	-0.0541	B-DNA	double	
60	Stacking energy	-17.5	-18.1	-15.8	-16.7	-19.5	-14.9	-19.2	-15.8	-14.7	-14.7	-14.9	-18.1	-17	-14.7	-19.5	-17.5	B-DNA	double	
61	Twist	35	32	28	31	43	35	31	28	41	40	35	32	43	41	43	35	B-DNA	double	
62	Tilt	0.1	-0.3	0.2	0.3	0	0.1	0	0.2	0	0	0.1	-0.3	-1.4	0	0	0.1	B-DNA	double	
63	Roll	1.4	1.4	5.5	-1.2	-1.2	3.9	6.2	5.5	0.4	-6.8	3.9	1.4	-0.6	0.4	-1.2	1.4	B-DNA	double	
64	Shift	-0.06	0.06	0.06	0.12	0.02	0.05	0.06	0.06	0	-0.3	0.05	0.06	-0.17	0	0.02	-0.06	B-DNA	double	
65	Slide	-0.16	-0.43	0.34	-0.57	1.88	0.28	0.68	0.34	-0.01	0.31	0.28	-0.43	0.38	-0.01	1.88	-0.16	B-DNA	double	
66	Rise	3.28	3.23	3.27	3.3	3.32	3.4	3.25	3.27	3.43	3.57	3.4	3.23	3.37	3.43	3.32	3.28	B-DNA	double	
67	Slide stiffness	2.26	3.03	2.03	3.83	1.78	1.65	2	2.03	1.93	2.61	1.65	3.03	1.2	1.93	1.78	2.26	B-DNA	double	
68	Shift stiffness	1.69	1.32	1.46	1.03	1.07	1.43	1.08	1.46	1.32	1.2	1.43	1.32	0.72	1.32	1.07	1.69	B-DNA	double	
69	Roll stiffness	0.02	0.023	0.019	0.022	0.017	0.019	0.016	0.019	0.02	0.026	0.019	0.023	0.016	0.02	0.017	0.02	B-DNA	double	
70	Tilt stiffness	0.038	0.038	0.037	0.036	0.025	0.042	0.026	0.037	0.038	0.036	0.042	0.038	0.018	0.038	0.025	0.038	B-DNA	double	
71	Twist stiffness	0.026	0.036	0.031	0.033	0.016	0.026	0.014	0.031	0.025	0.025	0.026	0.036	0.017	0.025	0.016	0.026	B-DNA	double	
72	Free energy	-1.2	-1.5	-1.5	-0.9	-1.7	-2.1	-2.8	-1.5	-1.5	-2.3	-2.1	-1.5	-0.9	-1.5	-1.7	-1.2	B-DNA	double	
73	Free energy	-1	-1.44	-1.28	-0.88	-1.45	-1.84	-2.17	-1.28	-1.3	-2.24	-1.84	-1.44	-0.58	-1.3	-1.45	-1	B-DNA	double	
74	Free energy	-1.02	-1.43	-1.16	-0.9	-1.7	-1.77	-2.09	-1.16	-1.46	-2.28	-1.77	-1.43	-0.9	-1.46	-1.7	-1.02	B-DNA	double	
75	Free energy	-0.91	-1.25	-1.28	-0.83	-1.54	-1.85	-1.87	-1.28	-1.3	-1.86	-1.85	-1.25	-0.68	-1.3	-1.54	-0.91	B-DNA	double	
76	GC content	0	1	1	0	1	2	2	1	1	2	2	1	0	1	1	0	DNA/RNA	single	
77	Purine (AG) content	2	1	2	1	1	0	1	0	2	1	2	1	1	0	1	0	DNA/RNA	single	
78	Keto (GT) content	0	0	0	1	0	0	1	1	1	1	2	2	1	1	1	2	DNA/RNA	single	
79	Adenine content	2	1	1	1	1	0	0	0	1	0	0	0	1	0	0	0	DNA/RNA	single	
80	Guanine content	0	0	1	0	0	0	1	0	1	1	2	1	0	0	1	0	DNA/RNA	single	
81	Cytosine content	0	1	0	0	1	2	1	1	0	1	0	0	0	1	0	0	DNA/RNA	single	
82	Thymine content	0	0	0	1	0	0	0	1	0	0	0	1	1	1	1	2	DNA/RNA	single	
83	Tilt (DNA-protein complex)	-1.4	-0.1	-1.7	0	0.5	-0.1	0	-1.7	-1.5	0	-0.1	-0.1	0	-1.5	0.5	-1.4	B-DNA	double	
84	Roll (DNA-protein complex)	0.7	0.7	4.5	1.1	4.7	3.6	5.4	4.5	1.9	0.3	3.6	0.7	3.3	1.9	4.7	0.7	B-DNA	double	
85	Shift (DNA-protein complex)	-0.03	0.13	0.09	0	0.09	0.05	0	0.09	-0.28	0	0.05	0.13	0	-0.28	0.09	-0.03	B-DNA	double	
86	Slide (DNA-protein complex)	-0.08	-0.58	-0.25	-0.59	0.53	-0.22	0.41	-0.25	0.09	-0.38	-0.22	-0.58	0.05	0.09	0.53	-0.08	B-DNA	double	
87	Rise (DNA-protein complex)	3.27	3.36	3.34	3.31	3.33	3.42	3.39	3.34	3.37	3.4	3.42	3.36	3.42	3.37	3.33	3.27	B-DNA	double	
88	Twist	35.8	35.8	30.5	33.4	36.9	33.4	31.1	30.5	39.3	38.3	33.4	35.8	40	39.3	36.9	35.8	B-DNA	double	
89	Tilt	-0.4	-0.9	-2.6	0	0.6	-1.1	0	-2.6	-0.4	0	-1.1	-0.9	0	-0.4	0.6	-0.4	B-DNA	double	
90	Roll	0.5	0.4	2.9	-0.6	1.1	6.5	6.6	2.9	-0.1	-7	6.5	0.4	2.6	-0.1	1.1	0.5	B-DNA	double	
91	Slide	-0.03	-0.13	0.47	-0.37	1.46	0.6	0.63	0.47	-0.07	0.29	0.6	-0.13	0.74	-0.07	1.46	-0.03	B-DNA	double	
92	Twist	35.3	32.6	31.2	31.2	39.2	33.3	36.6	31.2	40.3	37.3	33.3	32.6	40.5	40.3	39.2	35.3	B-DNA	double	
93	Tilt	0.5	0.1	2.8	0	-0.7	2.7	0	2.8	0.9	0	2.7	0.1	0	0.9	-0.7	0.5	B-DNA	double	
94	Roll	0.3	0.5	4.5	-0.8	0.5	6	3.1	4.5	-1.3	-6.2	6	0.5	2.8	-1.3	0.5	0.3	B-DNA	double	
95	Shift	0	0.2	-0.4	0	0.1	0	0	-0.4	0	0	0	0.2	0	0	0.1	0	B-DNA	double	
96	Slide	-0.1	-0.2	0.4	-0.4	1.6	0.8	0.7	0.4	0	0.4	0.8	-0.2	0.9	0	1.6	-0.1	B-DNA	double	
97	Rise	3.3	3.3	3.3	3.3	3.4	3.4	3.4	3.3	3.3	3.5	3.4	3.3	3.4	3.3	3.4	3.3	B-DNA	double	
98	Twist	35.62	34.4	27.7	31.5	34.5	33.67	29.8	27.7	36.9	40	33.67	34.4	36	36.9	34.5	35.62	B-DNA	double	
99	Wedge	7.2	1.1	8.4	2.6	3.5	2.1	6.7	8.4	5.3	5	2.1	1.1	0.9	5.3	3.5	7.2	B-DNA	double	
100	Direction	-154	143	2	0	-64	-57	0	-2	120	180	57	-143	0	-120	64	154	DNA	single	
101	Slide (RNA)	-1.27	-1.43	-1.5	-1.36	-1.46	-1.78	-1.89	-1.5	-1.7	-1.39	-1.78	-1.43	-1.45	-1.7	-1.46	-1.27	A-RNA	double	
102	Rise (RNA)	3.18	3.24	3.3	3.24	3.09	3.32	3.3	3.3	3.38	3.22	3.32	3.24	3.26	3.38	3.09	3.18	A-RNA	double	
103	Tilt (RNA)	-0.8	0.8	0.5	1.1	1	0.3	-0.1	0.5	1.3	0	0.3	0.8	-0.2	1.3	1	-0.8	A-RNA	double	
104	Roll (RNA)	7	4.8	8.5	7.1	9.9	8.7	12.1	8.5	9.4	6.1	12.1	4.8	10.7	9.4	9.9	7	A-RNA	double	
105	Twist (RNA)	31	32	30	33	31	32	27	30	32	35	32	32	32	32	31	31	A-RNA	double	
106	Stacking energy (RNA)	-13.7	-13.8	-14	-15.4	-14.4	-11.1	-15.6	-14	-14.2	-16.9	-11.1	-13.8	-16	-14.2	-14.4	-13.7	A-RNA	double	
107	Rise stiffness	7.65	8.93	7.08	9.07	6.38	8.04	6.23	7.08	8.56	9.53	8.04	8.93	6.23	8.56	6.38	7.65	B-DNA	double	
108	Melting Temperature	0.945	1.07	0.956	0.952	0.945	1.036	0.997	0.956	1.037	1.18	1.036	1.07	0.894	1.037	0.945	0.945	B-DNA	double	
109	Stacking energy	0.703	1.323	0.78	0.854	0.79	0.984	1.124	0.78	1.23	1.792	0.984	1.323	0.615	1.23	0.79	0.703	B-DNA	double	
110	Enthalpy (RNA)	-6.6	-10.2	-7.6	-5.7	-10.5	-12.2	-8	-7.6	-13.3	-14.2	-12.2	-10.2	-8.1	-10.2	-7.6	-6.6	A-RNA	double	
111	Entropy (RNA)	-18.4	-26.2	-19.2	-15.5	-27.8	-29.7	-19.4	-19.2	-35.5	-34.9	-29.7	-26.2	-22.6	-26.2	-19.2	-18.4	A-RNA	double	
112	Free energy (RNA)	-0.9	-2.1	-1.7	-0.9	-1.8	-2.9	-2	-1.7	-2.3	-3.4	-2.9	-2.1	-1.1	-2.1	-1.7	-0.9	A-RNA	double	
113	Free energy (RNA)	-0.93	-2.24	-2.08	-1.1	-2.11	-3.26	-2.36	-2.08	-2.35	-3.42	-3.26	-2.24	-1.33	-2.35	-2.11	-0.93	A-RNA	double	
114	Enthalpy (RNA)	-6.82	-11.4	-10.48	-9.38	-10.44	-13.39	-10.64	-10.48	-12.44	-14.88	-13.39	-11.4	-7.69	-12.44	-10.44	-6.82	A-RNA	double	
115	Entropy (RNA)	-19	-29.5	-27.1	-26.7	-26.9	-32.7	-26.7	-27.1	-32.5	-36.9	-32.7	-29.5	-20.5	-32.5	-26.9	-19	A-RNA	double	
116	Roll	-5.4	-2.5	1	-7.3	6.8	1.3	4.6	1	2	-3.7	1.3	-2.5	8	2	6.8	-5.4	B-DNA	single	
117	Tilt	-0.5	-2.7	-1.6	0	0.4	0.6	0	1.6	-1.7	0	-0.6	2.7	0	1.7	-0.4	0.5	B-DNA	single	
118	Twist	36	33.7	34.4	35.3	34.1	33.1	33.5	34.4	34.6	33.3	33.1	33.7	34.5	34.6	34.1	36	B-DNA	single	
119	Roll	2.3	-2	0.5	-8.1	7.4	1.4	6.3	0.5	5	-0.4	1.4	-2	8.4	5	7.4	2.3	DNA	double	
120	Twist	37.6	35.8	35.7	39.7	32.2	35.5	33.9	35.7	38.4	37.4	35.5	35.8	34.6	38.4	32.2	37.6	DNA	double	
121	Flexibility_slide	13.72	9.57	7.58	11.69	1.35	7.36	4.02	7.58	10.28	4.34	7.36	9.57	7.13	10.28	1.35	13.72	DNA	double	
122	Flexibility_shift	5.35	9.73	8.98	1.13	4.61	5.51	12.13	8.98	5.44	1.98	5.51	9.73	4.28	5.44	4.61	5.35	DNA	double	
123	Enthalpy	-7.6	-8.4	-7.8	-7.2	-8.5	-8	-10.6	-7.8	-8.2	-9.8	-8	-8.4	-7.2	-8.2	-8.5	-7.6	DNA	double	
124	Entropy	-21.3	-22.4	-21	-20.4	-22.7	-19.9	-27.2	-21	-22.2	-24.4	-19.9	-22.4	-21.3	-22.2	-22.7	-21.3	DNA	double	
125	Free energy	-1	-1.44	-1.28	-0.88	-1.45	-1.84	-2.17	-1.28	-1.3	-2.24	-1.84	-1.44	-0.58	-1.3	-1.45	-1	DNA	double	""")

class StruM(object):
	"""docstring for StruM"""
	def __init__(self, load_diprodb=True, mode="full", n_process=None, custom_filter=None):
		super(StruM, self).__init__()
		self.n_process = n_process
		if self.n_process == -1:
			from multiprocessing import cpu_count
			self.n_process = cpu_count()
		self.k = None
		self.p = None
		self.strum = None
		self.diprodb_data = None
		self.data = None
		self.func = None
		self.PWM = None
		self.features = []
		self.mins = []
		self.scale = []
		if load_diprodb:
			import pandas as pd
			diprodb.seek(0)
			self.diprodb_data = pd.read_table(diprodb)
			N = len(self.diprodb_data)
			mask = np.array([False for i in range(N)])
			if mode == "basic":
				filter = [1, 3, 4]
				filter = [x-1 for x in filter]
			elif mode == "groove":
				filter = [7, 8, 9, 10, 11, 12, 13, 14]
				filter = [x-1 for x in filter]
			elif mode == "protein":
				filter = [25, 26, 27, 28, 30, 32]
				filter = [x-1 for x in filter]
			elif mode == "full":
				filter = np.array(range(N))
				filter = filter[np.asarray((self.diprodb_data["Strand"]=="double") & ((self.diprodb_data["NucleicAcid"]=="DNA") | (self.diprodb_data["NucleicAcid"]=="B-DNA")))]
			elif mode == "nucs":
				filter = [79, 80, 81, 82]
				filter = [x-1 for x in filter]
			elif mode == "unique":
				filter = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
				          14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 
				          26, 27, 28, 30, 32, 34, 67, 68, 69, 70, 71, 
				          76, 77, 78, 79, 80, 81, 82, 99, 100, 107]
				filter = [x-1 for x in filter]
			elif mode == "proteingroove":
				filter = [7, 8, 9, 10, 11, 12, 13, 14] + [25, 26, 27, 28, 30, 32]
				filter = [x-1 for x in filter]
			elif mode == "custom" or custom_filter is not None:
				filter = custom_filter
			else:
				raise NameError("Unknown mode '%s'. Pick from [basic, groove, protein, full, nucs, unique, custom]" % mode)
			mask[filter] = True
			self.diprodb_data = self.diprodb_data[pd.Series(mask)]
			for i in self.diprodb_data.ID:
				self.features.append(self.diprodb_data.loc[i-1][1])
				row = self.diprodb_data.loc[i-1][2:18]
				mean = np.mean(row)
				sd = np.std(row)
				self.mins.append(abs(min(row)))
				self.scale.append((mean, sd))
				self.diprodb_data.loc[i-1,2:18] = [(x-mean)/sd for x in row]

	def update(self, data=None, features=None, func=None):
		if data is not None:
			self.data = data
		if features is not None:
			for x in features:
				self.features.append(x)
		if func is not None:
			self.func = func
	def translate(self, seq, *args, **kwargs):
		diprodb = addition = None
		if self.diprodb_data is not None:
			diprodb = self.translate_diprodb(seq)
		if self.func is not None:
			addition = self.func(self.data, *args, **kwargs)
		if diprodb is not None:
			if addition is not None:
				return np.ravel(np.vstack([diprodb, addition]).T)
			else:
				return np.ravel(diprodb.T)
		else:
			return np.ravel(addition.T)
	def scaler(self, trace):
		out = []
		n = len(self.scale)
		for i,x in enumerate(trace):
			j = i%self.p
			if j >= n:
				mu = 0
				sd = 1
			else:
				mu, sd = self.scale[j]
			out.append(x*sd + mu)
		return out
	def translate_scale(self, seq, *args, **kwargs):
		return self.scaler(self.translate(seq, *args, **kwargs))
	def translate_diprodb(self, seq):
		row = []
		for i in range(len(seq)-1):
			di = seq[i:i+2]
			if 'N' in di:
				row.append([0.0]* len(self.diprodb_data['AA']))
			else:
				row.append(list(self.diprodb_data[di]))
		return np.asarray(row).T
	def train(self, training_sequences, weights=None, lim=None, **kwargs):
		if type(training_sequences[0]) != str:
			if type(training_sequences[0][0]) == str:
				self.k = len(training_sequences[0][0])
				args_pos = 1
		else:
			self.k = len(training_sequences[0])
			args_pos = None
		data = []
		just_sequences = []
		for example in training_sequences:
			if args_pos is 1:
				seq = example[0]
				args = example[args_pos]
			else:
				seq = example
				args = []
			assert len(seq) == self.k
			just_sequences.append(seq)
			data.append(self.translate(seq, *args, **kwargs))

		arr = np.asarray(data)
		if weights is None:
			weights = np.ones(arr.shape[0])
		average = np.average(arr, axis=0, weights=weights)
		self.p = average.shape[0]/(self.k-1)
		variance = np.average((arr-average)**2, axis=0, weights=weights)
		self.strum = [average, np.sqrt(variance)]
		if lim is not None:
			self.strum[1][self.strum[1] < lim] = lim
		self.define_PWM(just_sequences, weights=weights)
	def norm_pdf(self, x, mu, var):
		#result = (1./np.sqrt(2*np.pi*var))*np.exp(-1*(x-mu)**2/(2*var))
		result = ndtr(-np.absolute(x-mu)/var)
		result += 10**-300
		return result
	def eval(self, kmer):
		return np.sum(np.log10(self.norm_pdf(kmer, self.strum[0], self.strum[1]**2)))
	def score_seq(self, seq, *args, **kwargs):
		scores = []
		struc_seq = self.translate(seq, *args, **kwargs)
		for i in range(0, len(struc_seq) - (self.k - 1)*self.p + self.p, self.p):
			kmer = struc_seq[i:i + (self.k - 1)*self.p]
			scores.append(self.eval(kmer))
		return scores
	def rev_comp(self, seq):
		nucs = "ACGT"
		index = dict(zip(nucs, nucs[::-1]))
		index['N'] = 'N'
		return "".join([index[n] for n in seq][::-1])
	def plot(self, save_path):
		logo_vals = np.reshape(self.strum[0], [self.k-1, self.p]).T
		logo_wts  = np.reshape(self.strum[1], [self.k-1, self.p]).T
		new_names = self.features

		ranges = []
		for i in self.diprodb_data.ID:
			row = self.diprodb_data.loc[i-1][2:18]
			ranges.append([np.min(row), np.max(row)])

		n = logo_vals.shape[0]
		m = logo_vals.shape[1]
		xs = np.asarray(range(1,m+1))
		colors = ['darkorange']
		figwidth = 3+(m+1)/3.
		figheight = float(n)*(figwidth-3)/m
		plt.figure(figsize=[figwidth,figheight])
		override = {
		   'verticalalignment'   : 'center',
		   'horizontalalignment' : 'right',
		   'rotation'            : 'horizontal',
		   #'size'                : 22,
		   }

		for i in range(n):
			plt.subplot(n,1,i+1)
			up = logo_vals[i] + logo_wts[i]
			dn = logo_vals[i] - logo_wts[i]
			plt.plot(xs, logo_vals[i], color='black', zorder=10)
			y1, y2 = plt.ylim()
			plt.fill_between(xs, up, dn, alpha=0.2, color=colors[i%len(colors)], zorder=1)
			#plt.plot(xs, up, xs, dn, color='black', zorder=5)
			plt.xticks([])
			plt.yticks([])
			plt.xlim([xs[0],xs[-1]])
			plt.ylim(ranges[i])
			plt.ylabel(new_names[i], **override)

		plt.xticks(range(1,m+1))
		plt.xlabel("Position")

		plt.tight_layout()
		plt.savefig(save_path, dpi=400)
		plt.close()
	def read_FASTA(self, fasta_file):
		sequences = []
		headers = []
		header = None
		seq = ""
		for line in fasta_file:
			if line.startswith(">"):
				if header is None:
					header = line.strip()[1:]
				else:
					headers.append(header)
					sequences.append(seq)
					header = line.strip()[1:]
					seq = ""
			else:
				seq += line.strip()
		headers.append(header)
		sequences.append(seq)
		return headers, sequences
	def train_EM(self, data, fasta=True, params=None, k=10, max_iter=1000, 
		         convergence_criterion=0.001, random_seed=None, 
		         n_init=1, lim=None):
		import sys
		global back_logL, match_motif, back_motif
		global II, K, p, sequences_data, M, max_motif, match_motif_denom

		if self.n_process is None:
			class Pool(object):
				"""docstring for Pool"""
				def __init__(self, arg):
					super(Pool, self).__init__()
					self.arg = arg
				def map(self, func, array):
					return [func(thing) for thing in array]
				def join(self):
					return
				def close(self):
					return
		else:
			from multiprocessing import Pool
			#from multiprocessing.dummy import Pool
					
		import random
		if random_seed is not None:
			random.seed(random_seed)

		if fasta:
			headers, sequences = self.read_FASTA(data)
		else:
			sequences = data
		sequences = [seq.upper() for seq in sequences]
		sequences_data = []
		if params is None:
			for s in sequences:
				sequences_data.append( self.translate(s) )
				sequences_data.append( self.translate(self.rev_comp(s)) )
		else:
			for i, s in enumerate(sequences):
				sequences_data.append(self.translate(s, *params[i]))
		def cleanM(M):
			pops = []
			II = []
			for i in range(0,len(M),2):
				m1 = np.max(M[i])
				m2 = np.max(M[i+1])
				if m1 > m2: 
					pops.append(i+1) 
					II.append(i)
				else:
					pops.append(i)
					II.append(i+1)

			pops.sort(reverse=True)
			for i in pops: M.pop(i)
			return M, II

		# User random restarts to compensate for local maxima in the landscape
		restart_vals = []
		for i in range(n_init):
			print >> sys.stderr, "Initializing motifs"
			K = k
			self.k = k + 1
			self.p = len(self.features)
			p = self.p

			## Initialize background 'motif'
			back_stuff = [[] for i in range(p)]
			for s in sequences_data:
				for i in range(len(s)):
					back_stuff[i%p].append(s[i])

			back_motif = [[np.average(x),np.std(x)] for x in back_stuff]
			back_motif = [[0.,1.] for i in range(p)]
			del back_stuff

			## Initialize match 'motif' randomly
			match_motif = [[random.random(),0.5] for i in range(k*p) ]
			#match_motif = zip(lookup(create_random_OO(k+1,"C",0)),[0.5 for i in range(k*p)])

			max_motif = [x[0] for x in match_motif]
			motif_error = [x[1] for x in match_motif]

			LIKELIHOODS = []

			print >> sys.stderr, "Prepping background likelihood"
			# Calculate log likelihood for each sequence matching background:
			back_logL = []
			for s in sequences_data:
				logL = 0.
				for i in range(len(s)):
					logL += np.log( self.norm_pdf(s[i], back_motif[i%p][0], back_motif[i%p][1]))
				back_logL.append(logL)

			back_motif_avg = np.asarray([x[0] for x in back_motif])
			back_motif_std = np.asarray([x[1] for x in back_motif])
			back_motif = [back_motif_avg, back_motif_std]

			print >> sys.stderr, "Starting Expectation maximization"
			lastlogL = None
			lastM = None
			cycle = False

			for __ in range(max_iter):
				print >> sys.stderr, ".",
				# Do Expectation step, once
				## Given the motif above, what is the probability of seeing each kmer? P(X|theta)
				M = []
				match_motif_avg = np.asarray([x[0] for x in match_motif])
				match_motif_std = np.asarray([x[1] for x in match_motif])
				match_motif = [match_motif_avg, match_motif_std]
				
				pool = Pool(self.n_process)
				M = pool.map(EM_wrap0, enumerate(sequences_data))
				pool.close()
				pool.join()

				M, II = cleanM(M)


				logL = 0.
				#  Normalize the values based on the rest of the row
				## Because the values in M should be P_i / sum(P_j for j in range(len(row)))
				## In log space: log(P_i) - log(sum(P_j))
				## Using an identity: log(sum(P_j)) = P_0 + np.log(1+np.sum([ np.exp(P_j-P_0) for j in [1:] ]))
				## Then I convert it to normal space, now that we are avoiding underflow issues
				for i in range(len(M)):
					#M[i] = [1 if x==np.max(M[i]) else 0 for x in M[i]]
					tmp = sorted(M[i], reverse=True)
					denom = tmp[0] + np.log(1+np.sum([np.exp(x-tmp[0]) for x in tmp[1:]]))
					logL += denom
					M[i]  = [np.exp(x-denom) for x in M[i]]

				# Check convergence
				if logL in LIKELIHOODS:
					if cycle:
						if logL == cycle_max:
							print >> sys.stderr, "\nStopped after %d iterations" % (__ + 1)
							break
					else:
						print >> sys.stderr, "\nDetected cyclical likelihoods. Proceeding to max..."
						for i,l in enumerate(LIKELIHOODS):
							if l == logL: 
								IN = i
								#cycle_size = len(LIKELIHOODS) - (i+1)
								cycle = True
								break
						cycle_max = np.max(LIKELIHOODS[i:])


				#logL = np.sum([np.product(x) for x in M])
				LIKELIHOODS.append(logL)
				#logL /= abs(LIKELIHOODS[0])
				if lastlogL:
					if abs(logL-lastlogL) < convergence_criterion: #0.001:
						print >> sys.stderr, "\nConverged after %d iterations based on likelihood" % (__ + 1)
						break
				lastlogL = logL

				#if not checkTime(start_time, time.time(), max_time):
				#	print "\nApproaching time, stopped after %d iterations" % (__ + 1)
				#	break

				# Do Maximum Likelihood estimation of motif
				match_motif_denom = [np.sum([np.sum(x) for x in M]), np.sum([np.sum(np.square(x)) for x in M])]

				# prep = []
				# for jj in range(len(II)):
				# 	ii = II[jj]
				# 	s = sequences_data[ii]
				# 	prep.append([jj,k,p,ii,s,M[jj],match_motif_denom[0]])

				pool = Pool(self.n_process)
				max_motif = pool.map(EM_wrap1,range(len(II)))#prep)
				pool.close()
				pool.join()
				
				max_motif = list(np.sum(max_motif,axis=0))

				# del prep
				# prep = []
				# for jj in range(len(II)):
				# 	ii = II[jj]
				# 	prep.append([jj,k,p,ii,sequences_data[ii],M[jj],max_motif,match_motif_denom])

				pool = Pool(self.n_process)
				motif_error = pool.map(EM_wrap2,range(len(II)))# prep)
				pool.close()
				pool.join()

				motif_error = list(np.sum(motif_error,axis=0))

				match_motif = [list(x) for x in zip(max_motif,motif_error)]

				thresh = 0.0001 # 10.0**-150 # 0.001
				for i in range(len(match_motif)):
					if match_motif[i][1] < thresh: match_motif[i][1] = thresh

				lastM = M

				max_motif = [x[0] for x in match_motif]
				motif_error = [x[1] for x in match_motif]


			if __ == max_iter - 1:
				print >> sys.stderr, "\nDid not converge after %d iterations" % max_iter

				match_motif_avg = np.asarray([x[0] for x in match_motif])
				match_motif_std = np.asarray([x[1] for x in match_motif])
				match_motif = [match_motif_avg, match_motif_std]



			if len(M) == len(sequences_data):
				M,II = cleanM(M)
				logL = 0.
				#  Normalize the values based on the rest of the row
				## Because the values in M should be P_i / sum(P_j for j in range(len(row)))
				## In log space: log(P_i) - log(sum(P_j))
				## Using an identity: log(sum(P_j)) = P_0 + np.log(1+np.sum([ np.exp(P_j-P_0) for j in [1:] ]))
				## Then I convert it to normal space, now that we are avoiding underflow issues
				for i in range(len(M)):
					#M[i] = [1 if x==np.max(M[i]) else 0 for x in M[i]]
					tmp = sorted(M[i], reverse=True)
					denom = tmp[0] + np.log(1+np.sum([np.exp(x-tmp[0]) for x in tmp[1:]]))
					logL += denom
					M[i]  = [np.exp(x-denom) for x in M[i]]

			restart_vals.append((match_motif, logL, M, II))
			print LIKELIHOODS

		restart_vals.sort(key=lambda x:x[1], reverse=True)
		print "Restart Likelihoods:", [x[1] for x in restart_vals]
		match_motif, logL, M, II = restart_vals[0]
		self.strum = match_motif
		if lim is not None:
			self.strum[1][self.strum[1] < lim] = lim

		pwm_seqs = []
		for i in range(len(M)):
			toggle = II[i] % 2
			n = np.argmax(M[i])
			s = sequences[i]
			if toggle == 1: s = self.rev_comp(s)
			pwm_seqs.append(s[n:n+k+1])

		self.define_PWM(pwm_seqs)

	def define_PWM(self, seqs, weights=None):
		nuc_index = dict(zip("ACGT", range(4)))
		if weights is None:
			weights = [1.0] * len(seqs)
		pwm = np.zeros([4,self.k])
		for i, seq in enumerate(seqs):
			for j, n in enumerate(seq):
				if n == "N": continue
				pwm[nuc_index[n], j] += weights[i]
		pwm /= np.sum(pwm, axis=0)
		self.PWM = pwm

	def print_PWM(self, labels=False):
		nuc_index = dict(zip("ACGT", range(4)))
		rows = [ " ".join(["%0.3f" % x for x in row]) for row in self.PWM ]
		if labels:
			for n in nuc_index:
				rows[nuc_index[n]] = n + " " + rows[nuc_index[n]]
			header = [" ".join([' '*(5-len(x)) + x for x in [str(i+1) for i in range(self.k)]])]
			rows = header + rows
		pretty = "\n".join(rows)
		print pretty
		return pretty

def EM_wrap0(ns):
	n = ns[0]
	s = ns[1]
	M_row = []
	#M.append([])
	logL_upto = 0.
	logL_after = back_logL[n]
	for i in range(0,len(s) - K*p + 1,p):
		kmer = s[i:i+K*p]
		logL = np.sum(np.log(norm_pdf(kmer,match_motif[0],match_motif[1])))

		change_back = 0.
		if i == 0:
			for j in range(0,K*p,p):
				change_back += np.sum(np.log(norm_pdf(kmer[j:j+p],back_motif[0],back_motif[1])))
		
		change_up = np.sum(np.log(norm_pdf(kmer[:p],back_motif[0],back_motif[1])))
		change_down = np.sum(np.log(norm_pdf(kmer[-p:],back_motif[0],back_motif[1])))
		
		if i == 0: logL_after -= change_back
		else: logL_after -= change_down
		
		#M[-1].append(logL_upto + logL + logL_after)
		M_row.append(logL_upto + logL + logL_after)
		logL_upto += change_up
	return M_row

def EM_wrap1(jj):
	adjustment = [0. for i in range(K*p)]
	ii = II[jj]
	s = sequences_data[ii]
	for i in range(0,len(s) - K*p + 1,p):
		for j in range(i,i+K*p):
			n = j-i
			adjustment[n] += s[j] * M[jj][i//p] / match_motif_denom[0]
	return adjustment
def EM_wrap2(jj):
	adjustment = [0. for i in range(K*p)]
	ii = II[jj]
	s = sequences_data[ii]
	for i in range(0,len(s) - K*p + 1,p):
		for j in range(i,i+K*p):
			n = j-i
			adjustment[n] += M[jj][i//p]*(s[j]-max_motif[n])**2/(match_motif_denom[0]-(match_motif_denom[1]/match_motif_denom[0]))
	return adjustment

def norm_pdf(x, mu, var):
	# result = (1./np.sqrt(2*np.pi*var))*np.exp(-1*(x-mu)**2/(2*var))
	result = ndtr(-np.absolute(x-mu)/var)
	result += 10**-300
	return result