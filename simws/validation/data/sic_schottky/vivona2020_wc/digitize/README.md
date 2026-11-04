How iv.csv was made (reproducible from the paper PDF, which is not stored here):
1. Extract the Fig. 4 JPEG from arXiv:2007.04038v1 page 5 (PyMuPDF: page.get_images, the 659x627 image) as p4_1.jpeg.
2. python digit.py   -> raw.json  (colour separation; axes: V = (x - 121.08)/338.54, log10 J = -11 + (534.5 - y)/39.04)
3. python clean.py   -> iv.csv    (artefacts removed by a running median, 10 mV resampling)
Checked by overlaying iv.csv on the figure; digitization uncertainty about +-1 px (+-6 % in J, +-3 mV in V).
