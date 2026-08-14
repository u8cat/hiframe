Sources and copyright conditions of logos (public domain if unspecified):

- default.svg: https://unsplash.com/illustrations/a-drawing-of-a-camera-on-a-white-background-p_Lvm8TJCZI Unsplash License
- apple.1977-04-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Apple_Computer_Logo_rainbow.svg
- apple.1998-08-15T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Apple_logo_black.svg
- canon.1956-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Canon_wordmark.svg
- dji.2009-01-01T00:00:00.svg: https://en.wikipedia.org/wiki/File:DJI_Innovations_logo.svg
- fujifilm.1934-01-01T00:00:00.svg: traced here from
    https://1000logos.net/wp-content/uploads/2020/04/Fujifilm-Logo-history.jpg
- fujifilm.1960-01-01T00:00:00.svg: traced here from the same history image
- fujifilm.1980-01-01T00:00:00.svg: traced here from the same history image
    None of the three marks Fuji Photo Film carried before 1992 is published as a
    drawing. The company shows all of them on
    https://holdings.fujifilm.com/en/about/brand/logo, at 600 by 400, and
    1000logos shows the same three in one image at 1001 by 1197, where each is
    the larger of the two, so the traces were taken from that and checked against
    the renditions of the company itself. The two agree with each other to 95%
    of their union for the box of 1980, 88% for the ellipse of 1960 and 83% for
    the circle of 1934, the thinner the strokes the worse, as befits two
    rasterizations at different sizes; each trace lands within a point of that
    ceiling, so the traces are as close to the company's own images as those
    images are to each other.
    The coverage of the ink was thresholded and run through
    `potrace -s --tight`, with `-a 0.5` for the box of 1980, whose edges are
    straight, and `-a 0.7` for the other two, from a four times upscaled copy
    except for the box, where tracing at the size of the source came out both
    smaller and no less faithful. The box is drawn in the #e53517 of the 1992
    file rather than the #e7330e measured on the JPEG of the company, the
    difference being within its compression noise, and the circle of 1934 in the
    warm near black #231816 that image carries rather than in pure black.
- fujifilm.1992-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Fujifilm_old_logo.svg
- fujifilm.2006-10-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Fujifilm_logo.svg
- google.2015-09-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Google_%22G%22_logo.svg
- google.2025-05-12T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Google_Favicon_2025.svg
- gopro.2002-01-01T00:00:00.svg: https://logos-download.com/brands/gopro
- gopro.2010-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:GoPro_logo_light.svg
- hasselblad.1941-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Hasselblad_logo.svg
- honor.2020-04-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Honor_Logo_(2020).svg
- honor.2026-07-23T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Honor_2026_logo.svg
- huawei.2006-05-08T00:00:00.svg: https://www-file.huawei.com/-/media/corporate/pdf/annual-report/annual_report2018_en_v2.pdf?la=en
    Page 1 of the 2018 annual report of Huawei still carries the logo of 2006,
    which no rendition on Wikimedia depicts. The page was cropped with
    `gs -sDEVICE=pdfwrite -dDEVICEWIDTHPOINTS=62 -dDEVICEHEIGHTPOINTS=63
    -dFIXEDMEDIA -c "<</PageOffset [-497 -37.5]>> setpagedevice"`, converted to
    SVG with `pdftocairo -svg`, and its viewBox tightened onto the mark.
    NOTE ON RIGHTS: unlike the logos above, this one is not claimed to be in
    the public domain. The eight petals with their gradients are far from the
    "simple geometric shapes or text" that puts a wordmark below the threshold
    of originality, and the file comes from a document of the company itself
    rather than from Commons. Review before redistributing.
- huawei.2018-03-27T00:00:00.svg: https://en.wikipedia.org/wiki/File:Huawei_Standard_logo.svg Public Domain in the United States
    Exported with `inkscape --export-type=svg --export-plain-svg` and then
    optimized with `python3 -m scour.scour --enable-comment-stripping
    --remove-descriptive-elements --no-line-breaks`, taking it from 816048 to
    26960 bytes. The drawing itself is only some 44000 bytes of path data; the
    rest was editor metadata, indentation and a style attribute on every
    element. Rendered at 400 px the two files differ in 19 pixels, all of them
    on the edge of a path, that is, in antialiasing alone.
    Note that Inkscape reads a colon in a file name as the separator of a URI
    scheme, so a logo here has to be copied to a name without one before
    Inkscape will open it.
- insta360.2015-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Insta360_logo_(transparent).svg
- iqoo.2019-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Iqoo.svg
- kodak.1971-01-01T00:00:00.svg: https://logos-download.com/brands/kodak
- kodak.1987-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Logo_of_the_Eastman_Kodak_Company_(1987%E2%80%932006).svg
- kodak.2006-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Logo_of_the_Eastman_Kodak_Company_(2006%E2%80%932016).svg
- kodak.2016-10-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Eastman_Kodak_Company_logo_(2016).svg
- leica.1976-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Leitz_Camera.svg
- leica.1986-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Leica_Camera.svg
- meizu.2006-01-01T00:00:00.svg: https://logos-download.com/brands/meizu
    Optimized with `python3 -m scour.scour --enable-comment-stripping
    --remove-descriptive-elements --no-line-breaks -p 10`, which took it from
    5067 to 1697 bytes, all of it editor metadata, and given the viewBox it
    lacked, `0 0 1000 174.93121`, matching its width and height. At full
    precision, `-p 10`, the drawing survives the pass untouched: rendered 600 px
    wide the two files agree in every pixel. The default precision of five
    digits instead moved edges by a fraction of a pixel.
- meizu.2015-09-22T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Meizu_logo2.svg
- meizu.2020-05-08T00:00:00.svg: the file for 2015 above, with its blue #21bbef replaced by #008cff
    The only change of 2020 was the brand color, so the drawing is that of
    2015 recolored, rather than a separate source.
- meizu.2023-03-08T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Meizu.svg
- motorola.1955-06-19T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Motorola_M_symbol_black.svg
- motorola.2011-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Motorola_Mobility.svg
    Cropped to the disc by narrowing the viewBox onto it, dropping the
    MOTOROLA MOBILITY wordmark beside it.
- motorola.2013-06-01T00:00:00.svg: drawn here, measured from
    https://commons.wikimedia.org/wiki/File:Motorola_logo_2013.png
    No drawing of this mark is published, so it was built from its geometry:
    twelve arcs of thirty degrees each, running from 0.932 to 0.997 of the
    radius, in the twelve colors read off the reference, around the batwing of
    the file for 1955 at 96% of its size in grey. The ring radii, the angles of
    the segments, their colors and the size of the batwing were all measured
    from the reference; the result covers 94.7% of it, and the batwing alone
    98.1%, the remainder being the antialiasing of the edges.
- motorola.2016-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Motorola_M_symbol_blue.svg
- nikon.1988-01-01T00:00:00.svg: https://seeklogo.com/vector-logo/99546/nikon
    Converted to svg using `eps2svg` and manually clipped the white rectangle background.
- nikon.2003-04-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Nikon_Logo.svg
- nokia.1978-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Nokia_wordmark.svg
- nokia.2023-02-26T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Nokia_2023.svg
- nubia.2012-01-01T00:00:00.svg: traced here from
    https://whatthelogo.com/logo/nubia/275507
    No drawing of the mark this brand carried before 2020 is published, and the
    only rendition found is 300 by 75. The two inks were separated by colour and
    traced apart, each from an eight times upscaled copy of its coverage, with
    `potrace -a 1.0 -O 0.2`, then reassembled in the greys and reds measured on
    the reference, #717175 and #c5470c. Rendered back at 300 px the file agrees
    with the reference in all but nine pixels, 99.5% of their union.
    That this is the mark of that era was checked twice over: a Chinese writeup
    of the change of 2020 describes rings and dots forming the letters and calls
    the mark the "little red circle" being bidden farewell, and a photograph of a
    nubia telephone in the same article carries these letterforms.
- nubia.2020-04-13T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Nubia_logo.svg
- olympus.1970-01-01T00:00:00.svg: the file for 2001 below, with the yellow polygon removed
    The wordmark of 1970 is that of 2001 without the Opto-Digital Pattern added
    beneath it, so the drawing is derived rather than sourced separately. Both
    files also had a registered sign drawn in white below their right end, which
    was removed: invisible against the white of the frame, it still counted
    towards the bounding box and so had the logo drawn a quarter smaller.
- olympus.2001-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Olympus_Corporation_logo.svg
- oneplus.2013-12-01T00:00:00.svg: drawn here, measured from
    https://commons.wikimedia.org/wiki/File:Logo_entreprise_OnePlus.png
    The mark is nine rectangles and nothing else, so rather than tracing the
    reference its edges were measured on it to a hundredth of a pixel and found
    to sit on a grid of the stroke width: the frame is ten strokes square, the
    plus four by four centred on the right edge, and the right edge is broken for
    one stroke below it. The drawing is that grid, in the #eb0028 of the
    reference, and covers 99.8% of it, the rest being single pixel slivers along
    the edges.
- oneplus.2020-03-18T00:00:00.svg: https://commons.wikimedia.org/wiki/File:OP_LU_Reg_1L_RGB_red_copy-01.svg
    Cropped to the square at its left, dropping the wordmark, by narrowing the
    viewBox from `0 0 608.8 147.2` to `0 0 147.2 147.2`.
- oppo.2004-01-01T00:00:00.svg: https://upload.wikimedia.org/wikipedia/commons/archive/b/b8/20180224123700%21OPPO_Logo.svg
    The revision of 16 March 2014 of
    https://commons.wikimedia.org/wiki/File:OPPO_Logo.svg, which draws the mark
    of 2004 rather than the one of 2013 that replaced it in the file in 2018.
- oppo.2013-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:OPPO_logo.svg
- oppo.2019-04-10T00:00:00.svg: https://commons.wikimedia.org/wiki/File:OPPO_LOGO_2019.svg
- panasonic.1971-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Panasonic_Group_logo.svg
- pentax.1980-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Pentax_Logo.svg
- realme.2018-11-15T00:00:00.svg: traced here from
    https://1000logos.net/wp-content/uploads/2026/02/Realme-Logo-2018.png
    No drawing of this mark is published. The reference is 3840 px wide, the "R"
    alone 545 by 744, and the strokes are not of one width, 64 units across
    vertically against 56 horizontally, so the letter is a drawing rather than a
    grid and was traced instead of measured: the coverage of the grey over the
    yellow was thresholded and run through `potrace -s --tight -a 1.0 -O 0.2`,
    which covers 99.6% of the reference. Tracing a four times upscaled copy
    reaches 100.0% but takes eight times the path data for edges half a pixel
    apart on a 545 px letter, indistinguishable once rendered.
    The yellow field is a plain rectangle, so it is drawn as one, in the palette
    of the two realme files on Commons, #ffc915 and #676c73, rather than in the
    #ffc80f and #676c72 of that PNG. Its size follows the reference, where the
    letter sits with an equal margin of 223 units on all four sides.
- realme.2020-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Realme_logo_SVG.svg
- realme.2024-01-01T00:00:00.svg: the file for 2020 above, with its yellow
    rectangle removed
    The change of 2024 kept the letters and dropped the field they sat on, so
    the drawing is that of 2020 less one path, rather than a separate source.
- redmi.2019-01-10T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Redmi_by_Xiaomi_Logo.svg
    Cropped to the wordmark by keeping only its group, dropping the "by
    Xiaomi" line below it.
- redmi.2024-11-21T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Xiaomi_Redmi_Logo.svg
- ricoh.1986-02-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Ricoh_logo_1986.svg
- ricoh.2005-09-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Ricoh_logo_2005.svg
- samsung.1993-03-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Samsung_old_logo_before_year_2015.svg
- samsung.2015-06-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Samsung_wordmark.svg
- sony.1973-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Sony_logo.svg
- vivo.2009-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Vivo_Logo.svg GNU Free Documentation License 1.2 or later, and public domain
- vivo.2019-02-09T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Vivo_logo_2019.svg
- xiaomi.2010-04-06T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Xiaomi_logo.svg
- xiaomi.2021-03-30T00:00:00.svg: https://commons.wikimedia.org/wiki/File:Xiaomi_logo_(2021-).svg
- zte.1985-01-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:ZTE_logo.svg
- zte.2015-03-01T00:00:00.svg: https://commons.wikimedia.org/wiki/File:ZTE-logo.svg

These logos are post-processed by `tighten.py`, which narrows the viewBox of
each drawing onto its ink, so that no logo is drawn smaller than the others
for carrying a transparent margin.
