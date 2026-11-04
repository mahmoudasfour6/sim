How iv.csv was made (reproducible from the paper PDF, which is not stored here):
1. Extract the Fig. 3 JPEG (642x614) from arXiv:2102.08927 page 4 with PyMuPDF as p3_2.jpeg.
2. python digit.py -> raw.json (HSV colour classes per temperature; axes from the tick marks: 253.1 px/V,
   50.67 px per decade; the '25 C'/'115 C' labels, the arrow and the left-axis ticks are masked; only V < 0.98 V,
   left of the insets)
3. python clean.py -> iv.csv (running-median artefact removal, 10 mV resampling, 0.04-0.96 V)
Overlay checked by eye on all seven curves; uncertainty about +-1 px: +-0.02 decade (+-5 %) in J, +-4 mV in V.
