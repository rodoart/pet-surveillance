%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ArticuloAPA                                                                  %
% Class File                                                                   %
% Version 1.5 (17/july/2021)                                                   %
%                                                                              %
% Authors:                                                                     %
% Rodolfo Arturo González Trillo                                               %
%                                                                              %
% Class license:                                                               %
% LPPL v1.3c (http://www.latex-project.org/lppl)                               %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%------------------------------------------------------------------------------%
%                       CLASS DEFINITION AND PARAMETERS                        %
%------------------------------------------------------------------------------%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{articuloAPA}[2021/07/14 My custom article class whith APA]

% Load article
\providecommand{\baseclass}{article}
\LoadClass{\baseclass}

%------------------------------------------------------------------------------%
%                       CUSTOM NAME COMMANDS & METADATA                        %
%------------------------------------------------------------------------------%
\RequirePackage[bookmarks, colorlinks]{hyperref}



\NewDocumentCommand{\firstauthor}{m}{%
  \hypersetup{pdfauthor={#1}} 
  \newcommand{\atfirstauthor}{#1}
}%
\NewDocumentCommand{\secondauthor}{m}{\newcommand{\atsecondauthor}{#1}}
\NewDocumentCommand{\thirdauthor}{m}{\newcommand{\atthirdauthor}{#1}}
\NewDocumentCommand{\teacher}{m}{\newcommand{\atteacher}{#1}}

\DeclareDocumentCommand{\title}{m}{%
  \hypersetup{pdftitle={#1}}
  \newcommand{\attitle}{#1}
  \renewcommand{\@title}{#1}
}%

\DeclareDocumentCommand{\date}{m}{%
  \newcommand{\atdate}{#1}
  \renewcommand{\@date}{#1}
}%


\NewDocumentCommand{\course}{m}{%
  \hypersetup{pdfsubject={#1}}
  \newcommand{\atcourse}{#1}
}%

\NewDocumentCommand{\activity}{m}{\newcommand{\atactivity}{#1}}

\NewDocumentCommand{\keywords}{m}{%
  \hypersetup{pdfkeywords={#1}}
  \newcommand{\atkeywords}{#1}
}%
\NewDocumentCommand{\assignment}{m}{\newcommand{\atassignment}{#1}}

    
%------------------------------------------------------------------------------%
%                                    FONTS                                     %
%------------------------------------------------------------------------------%
 
\RequirePackage{fontspec}
\RequirePackage[spanish,es-nodecimaldot]{babel}
\RequirePackage{amsmath}
\RequirePackage{unicode-math}

\setmainfont[Path=fonts/,
  BoldItalicFont  = calibriz.ttf,
  BoldFont        = calibrib.ttf,
  ItalicFont      = calibrii.ttf
]{calibri.ttf}

\setsansfont[Path=fonts/,
  BoldItalicFont  = calibrii.ttf,
  BoldFont        = calibri.ttf,
  ItalicFont      = calibrili.ttf]{calibril.ttf}

\setmonofont{Consolas}

\defaultfontfeatures{Scale=MatchUppercase}
\setmathfont{FiraMath-Regular.otf}

\setmathfont[Path=fonts/,
  range={up,"002B,"002D,`±,`×,`·,`÷,`¬,`∂,`∆,`∕,`∞,`⌐}]{calibri.ttf}
\setmathfont[range={\nabla}]{XITSMath-Regular.otf}
\setmathfont[Path=fonts/, range=it]{calibrii.ttf}
\setmathfont[Path=fonts/, range=bfup]{calibrib.ttf}
\setmathfont[Path=fonts/, range=bfit]{calibriz.ttf}

\RequirePackage[babel]{csquotes}

\setoperatorfont\normalfont % For log, sin, cos, etc.

% Fake SC
\newlength\fake@f
\newlength\fake@c
\def\fakesc#1{%
  \begingroup%
  \xdef\fake@name{\csname\curr@fontshape/\f@size\endcsname}%
  \fontsize{\fontdimen8\fake@name}{\baselineskip}\selectfont%
  \uppercase{#1}%
  \endgroup%
}
\makeatother


% Light font command

\DeclareTextFontCommand{\textlf}{\sffamily}

%------------------------------------------------------------------------------%
%                           LINE SPACING & MARGINS                             %
%------------------------------------------------------------------------------%
\RequirePackage{setspace}

% Spacing
\onehalfspacing

\RequirePackage[left=3cm,right=2cm,top=2.5cm,bottom=2.5cm]{geometry}

% Column separation.
\setlength{\columnsep}{0.5cm}

% To impede overfull hboxes
\setlength{\emergencystretch}{10pt}
%------------------------------------------------------------------------------%
%                                    COLORS                                    %
%------------------------------------------------------------------------------%
\RequirePackage[table]{xcolor}
\definecolor{aqua}{RGB}{0,152,205}
\definecolor{aqua1}{RGB}{133,223,255}
\definecolor{aqua2}{RGB}{194,239,255}
\definecolor{aqua3}{RGB}{230,244,249}

\definecolor{mydarkgreen}{RGB}{135,172,66}
\definecolor{mygreen}{RGB}{169,215,83}

\definecolor{mydarkyellow}{RGB}{194,154,52}
\definecolor{myyellow}{RGB}{242,193,65}

\definecolor{mydarkorange}{RGB}{196,126,34}
\definecolor{myorange}{RGB}{245,158,42}

\definecolor{mydarkred}{RGB}{190,50,54}
\definecolor{myred}{RGB}{238,63,68}

\definecolor{mydarkhibuscus}{RGB}{157,52,86}
\definecolor{myhibiscus}{RGB}{196,65,107}

\definecolor{mylightgray}{RGB}{231,230,230}



%------------------------------------------------------------------------------%
%                                    LINKS                                     %
%------------------------------------------------------------------------------%
%\RequirePackage[bookmarks, colorlinks]{hyperref}
\hypersetup{%
  pdfpagemode={UseOutlines},
  bookmarksopen=true,
  bookmarksopenlevel=0,
  hypertexnames=false,
  colorlinks=true,% Set to false to disable coloring links
  citecolor=aqua,% The color of citations
  linkcolor=aqua,% The color of references to document elements (sections, figures, etc)
  urlcolor=aqua,% The color of hyperlinks (URLs)
  pdfstartview={FitV},
  unicode,
  breaklinks=true,
}%

\RequirePackage[all]{hypcap} %Hiperref te lleva a la parte superior de la imagen.

%------------------------------------------------------------------------------%
%                                  BIBLATEX                                    %
%------------------------------------------------------------------------------%
% Use the bibtex backend with the authoryear citation style (which resembles APA)
\RequirePackage[%
  style=apa,
  backend=biber,
  isbn=true,
  url=true,
  doi=true,
  natbib=true
  ]{biblatex}%
\addbibresource{../references/biblography.bib}

\DeclareDelimFormat*{finalnamedelim}
  {\ifnum\value{liststop}>2 \finalandcomma\fi\addspace{y}\space}

% the bibliography also needs another conditional, so we can't wrap
% everything up with just the two lines above
\DeclareDelimFormat[bib,biblist]{finalnamedelim}{%
  \ifthenelse{\value{listcount}>\maxprtauth}
    {}
    {\ifthenelse{\value{liststop}>2}
       {\finalandcomma\addspace\bibstring{and}\space}
       {\addspace{y}\space}}}

% this is a special delimiter to solve the bugs reported in
% https://tex.stackexchange.com/q/417648/35864
\DeclareDelimFormat*{finalnamedelim:apa:family-given}{%
  \ifthenelse{\value{listcount}>\maxprtauth}
    {}
    {\finalandcomma\addspace{y}\space}}


%------------------------------------------------------------------------------%
%                      WHOLE COLOR LINKING CITE COMMANDS                       %
%------------------------------------------------------------------------------%
\makeatletter
\newcommand*{\linkblx@startlink}[1]{%
  \blx@sfsave\hyper@natlinkstart{\the\c@refsection @#1}\blx@sfrest}
\newcommand*{\linkblx@startlinkentry}{%
  \linkblx@startlink{\abx@field@entrykey}}
\newcommand*{\linkblx@endlink}{%
  \blx@sfsave\hyper@natlinkend\blx@sfrest}

\DeclareCiteCommand{\cite}
  {}%
  {\DeclareFieldFormat{bibhyperref}{####1}%
   \linkblx@startlinkentry
   \ifnumequal{\value{citecount}}{1}
     {\usebibmacro{prenote}}
     {}%
   \usebibmacro{citeindex}%
   \usebibmacro{cite}%
   \ifnumequal{\value{citecount}}{\value{citetotal}}
     {\usebibmacro{postnote}}
     {}%
   \iflastcitekey{}{\multicitedelim}%
   \linkblx@endlink}
  {}
  {}

\DeclareCiteCommand*{\cite}
  {}%
  {\DeclareFieldFormat{bibhyperref}{####1}%
   \linkblx@startlinkentry
   \ifnumequal{\value{citecount}}{1}
     {\usebibmacro{prenote}}
     {}%
   \usebibmacro{citeindex}%
   \usebibmacro{citeyear}%
   \ifnumequal{\value{citecount}}{\value{citetotal}}
     {\usebibmacro{postnote}}
     {}%
   \iflastcitekey{}{\multicitedelim}%
   \linkblx@endlink}
  {}
  {}

\DeclareCiteCommand{\parencite}
  {}%
  {\DeclareFieldFormat{bibhyperref}{####1}%
   \linkblx@startlinkentry
   \iffirstcitekey{\bibopenparen}{}%
   \ifnumequal{\value{citecount}}{1}
     {\usebibmacro{prenote}}
     {}%
   \usebibmacro{citeindex}%
   \usebibmacro{cite}%
   \ifnumequal{\value{citecount}}{\value{citetotal}}
     {\usebibmacro{postnote}}
     {}%
   \iflastcitekey{\bibcloseparen}{\multicitedelim}%
   \linkblx@endlink}
  {}
  {}

\DeclareCiteCommand*{\parencite}
  {}%
  {\DeclareFieldFormat{bibhyperref}{####1}%
   \linkblx@startlinkentry
   \iffirstcitekey{\bibopenparen}{}%
   \ifnumequal{\value{citecount}}{1}
     {\usebibmacro{prenote}}
     {}%
   \usebibmacro{citeindex}%
   \usebibmacro{citeyear}%
   \ifnumequal{\value{citecount}}{\value{citetotal}}
     {\usebibmacro{postnote}}
     {}%
   \iflastcitekey{\bibcloseparen}{\multicitedelim}%
   \linkblx@endlink}
  {}
  {}


\DeclareCiteCommand{\textcite}
  {\boolfalse{cbx:parens}}
  {\DeclareFieldFormat{bibhyperref}{####1}%
   \linkblx@startlinkentry
   \usebibmacro{citeindex}%
   \iffirstcitekey
     {\setcounter{textcitetotal}{1}}
     {\stepcounter{textcitetotal}%
      \textcitedelim}%
   \usebibmacro{textcite}%
   \iflastcitekey
     {}
     {\ifbool{cbx:parens}
        {\bibcloseparen\global\boolfalse{cbx:parens}}
        {}}%
   \ifnumequal{\value{citecount}}{\value{citetotal}}
     {\usebibmacro{textcite:postnote}}
     {}%
   \linkblx@endlink}
  {}
  {}

\DeclareMultiCiteCommand{\parencites}{\parencite}{}
\makeatother

\newcommand{\Figura}[1]{\hyperref[#1]{Figura \ref*{#1}}}
\newcommand{\Tabla}[1]{\hyperref[#1]{Tabla \ref*{#1}}}
\newcommand{\Seccion}[1]{\hyperref[#1]{Sección \ref*{#1}}}
\newcommand{\figura}[1]{\hyperref[#1]{figura \ref*{#1}}}
\newcommand{\tabla}[1]{\hyperref[#1]{tabla \ref*{#1}}}
\newcommand{\seccion}[1]{\hyperref[#1]{sección \ref*{#1}}}
\newcommand{\codigo}[1]{\hyperref[#1]{código \ref*{#1}}}
\newcommand{\salida}[2]{\hyperref[#1]{\textit{output} \ref*{#2}}}
\newcommand{\continuacion}[3]{\hyperref[#1]{código \ref*{#2}#3}}

%------------------------------------------------------------------------------%
%                                   FIGURES                                    %
%------------------------------------------------------------------------------%
\RequirePackage{graphicx}
\RequirePackage{hypcap}
\RequirePackage{booktabs,caption}
\RequirePackage{subcaption}
\RequirePackage{placeins} % Para que no pasen las tablas.

% Specifies where to look for included images.
\graphicspath{{figures/}{./}{../figures/}}

% Sources bellow captions.
\newcommand{\source}[1]{\vspace{-8pt} \caption*{ \textbf{Fuente:} {#1}} }
\captionsetup{justification=centerlast, font=small, labelfont=bf}

% New command for a rule to be used under captions.
\newcommand{\decoRule}[1]{\rule{#1}{.4pt}} 

%------------------------------------------------------------------------------%
%                                  TABLES                                      %
%------------------------------------------------------------------------------%
\RequirePackage{tabulary}
\RequirePackage[flushleft]{threeparttable}
% Cambio de nombre:
\addto\captionsspanish{
\def\listtablename{\'Indice de tablas}%
\def\tablename{Tabla}}

\newlength\tablelength
\setlength\tablelength{0.4\textwidth}

\newcommand{\centered}[1]{\begin{tabular}{l} #1 \end{tabular}}
%------------------------------------------------------------------------------%
%                             HEADERS AND FOOTERS                              %
%------------------------------------------------------------------------------%
\RequirePackage{fancyhdr}
\RequirePackage{multirow}
\pagestyle{fancy}
\fancyhf{}
\chead{\footnotesize \atactivity}
\rfoot{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\fancypagestyle{noheader}{
   \fancyhead{}
   \renewcommand{\headrule}{\hrule height 0pt \vspace{0mm}\hrule height 0pt}
}

%------------------------------------------------------------------------------%
%                                 TITLE PAGE                                   %
%------------------------------------------------------------------------------%
\RequirePackage{multirow}
\RequirePackage{array, booktabs, arydshln,xcolor}
\RequirePackage{anyfontsize}

\newcommand\VRule[1][\arrayrulewidth]{\vrule width #1}

\renewcommand{\maketitle}{
  \begin{titlepage}
  % Text.
    \vspace*{-0.5cm}
    \begin{tcolorbox}[
        standard jigsaw,
        opacityback=0,
        opacityframe=0,
        width=0.99\textwidth,
        height=0.99\textheight,
        left={\dimexpr0.495\textwidth-0.35\paperwidth\relax},
        right={\dimexpr0.495\textwidth-0.35\paperwidth\relax},
        halign=flush center,
        valign=center,
        fontupper = \fontsize{12pt}{13.8pt}\selectfont,
        code={\singlespacing}
      ]
      % Universidad.
      \includegraphics[height=2.9cm]{logo_internet.png}\\
      \vspace{0.5cm}
      
      \textlf{\fontsize{24pt}{0pt}\selectfont Universidad Internacional de La Rioja}\\
      \vspace{0.2cm}
      % Escuela
      \textlf{\fontsize{20pt}{0pt}\selectfont Escuela Superior de Ingeniería y Tecnología}\\
      
      \vspace{4cm}
      % Maestría 
      \textlf{\fontsize{18pt}{0pt}\selectfont Maestría en Inteligencia Artificial}\\
     
      % Título
      \begin{tcolorbox}[
        enhanced,
        frame hidden,          
        width=0.7\paperwidth,
        top=0pt,
        bottom=0pt,
        left=15pt,
        right=15pt,
        halign=flush center,
        colback=white,
        %borderline north={3pt}{0pt}{aqua},
        %borderline south={3pt}{0pt}{aqua},
        fontupper = \fontsize{26pt}{29.9pt}\selectfont \color{aqua},
        code={\setstretch{1.15}}
      ]
        \attitle
      \end{tcolorbox}

      \vspace{6.7cm}
      
      %  Alumnos y fecha
      \begin{center}
        \sffamily
        \begin{tabular}{!{\color{mylightgray}\VRule[0.5pt]}p{8.2cm}
                        !{\color{mylightgray}\VRule[0.5pt]}p{6.7cm}
                        !{\color{mylightgray}\VRule[0.5pt]}}
         \arrayrulecolor{mylightgray}\specialrule{0.5pt}{0pt}{0pt}
         \multirow{3}{8.1cm}{Trabajo de innovación presentado por:} &      \atfirstauthor \\ 
         & \atsecondauthor \\ 
         & \atthirdauthor \\ 
         \arrayrulecolor{mylightgray}\specialrule{0.5pt}{0pt}{0pt}
         Fecha: & \atdate \\ 
         \arrayrulecolor{mylightgray}\specialrule{0.5pt}{0pt}{0pt}
        \end{tabular}
      \end{center}

    \end{tcolorbox}
    \clearpage
    \newpage
    \mbox{}
    \clearpage
    \newpage
    \cleardoublepage 

  \end{titlepage}
}
%------------------------------------------------------------------------------%
%                      SECTION WITHOUT NUMBERS ARE INDEXED                     %
%------------------------------------------------------------------------------%
\newcommand{\indexed}[2]{%
	\addcontentsline{toc}{#1}{#2}
	\expandafter\csname#1\endcsname*{#2}
}


%------------------------------------------------------------------------------%
%                                  LIST STYLE                                  %
%------------------------------------------------------------------------------%
\renewcommand{\labelitemi}{$\textcolor{aqua}{\bullet}$}

%------------------------------------------------------------------------------%
%                                  CODE BOXES                                  %
%------------------------------------------------------------------------------%
\RequirePackage{tcolorbox}
\tcbuselibrary{minted, most}

% the space reserved between for the ``In'' numbers and the code
\newlength\inwd
\setlength\inwd{0.75cm}
\newcounter{ipythcntr}

\tcbset{
  ipynb/.style={
    enlarge left by=\inwd,
    width=\linewidth-\inwd,
    enhanced,
    top=0pt,
    bottom=0pt,
    left=0pt,
    right=0pt,
  }
}

\tcbset{
  ipynbbox/.style={
    sharp corners,
    boxrule=0.4pt,
    colback=aqua3,
    frame hidden,
    %colframe=aqua,
  }
}

\tcbset{
  ipynboverlayin/.style={
    overlay={
      \node[
        anchor=north east,
        text width=\inwd,
        font=\small\color{aqua},
        inner ysep=6pt,
        inner xsep=0pt,
        outer sep=0pt
      ] 
      at (frame.north west)
      {[\theipythcntr]:};
    },
  },
}

\tcbset{
  ipynbminted/.style={
    listing only,
    listing engine=minted,
    minted language=python,
    minted style=friendly,
    minted options={
      fontsize=\scriptsize,
      breaklines=true
    }, 
  },
}

\newtcblisting[use counter=ipythcntr, number format=\arabic]{ipynbin}[1][]{
  ipynb,
  ipynbbox,
  ipynboverlayin,
  ipynbminted,
  #1,
}

\newtcblisting{ipynbincont}[2][]{
  ipynb,
  ipynbbox,
  overlay={
      \node[
        anchor=north east,
        text width=\inwd,
        font=\small\color{aqua},
        inner ysep=6pt,
        inner xsep=0pt,
        outer sep=0pt
      ] 
      at (frame.north west)
      {
        [\arabic{ipythcntr}#2]: \\
        \vspace{-7pt}{\centering\tiny \textit{cont.}}\\
      };
    },
  ipynbminted,
  #1,
}



\tcbset{
  ipynboverlayout/.style={
    overlay={
      \node[
        anchor=north east,
        text width=\inwd,
        font=\small\color{mydarkgreen},
        inner ysep=4pt,
        inner xsep=0pt,
        outer sep=0pt
      ] 
      at (frame.north west)
      {[\arabic{ipythcntr}]:};
    },
  },
 }

\tcbset{
  ipynbnobox/.style={
    standard jigsaw,
    opacityback=0,
    frame hidden,
  },
}

\newtcolorbox{ipynboutnocode}[1][]{
  ipynbnobox,
  ipynb,
  ipynboverlayout,
  top=-3pt,
  fontupper=\scriptsize\ttfamily,
  #1, 
}

\newtcblisting{ipynbout}[1][]{
  ipynbnobox,
  ipynb,
  ipynboverlayout,
  top=-3pt,
  fontupper=\scriptsize\ttfamily,
  listing only,
  listing engine=minted,
  minted language=text,
  minted options={
    fontsize=\scriptsize,
    breaklines=true
  },
  #1, 
}

\RequirePackage{setspace}
\renewcommand{\baselinestretch}{1.5}
\newtcolorbox{commentb}{%
  ipynbnobox, 
  ipynb,
  left=-4pt,
  fontupper={\small\linespread{0.66}\selectfont}
}

\newtcblisting[use counter=ipythcntr, number format=\arabic]{ipynbincommented}[2][]{
  ipynb,
  ipynbbox,
  ipynboverlayin,
  ipynbminted,
  listing above comment,
  middle=0pt,
  comment={
    \begin{commentb}
      #2
    \end{commentb}
  },
  #1,
}

\newtcblisting{ipynboutcommented}[2][]{
  ipynbnobox,
  ipynb,
  ipynboverlayout,
  top=-3pt,
  fontupper=\scriptsize\ttfamily,
  listing only,
  listing engine=minted,
  minted language=text,
  minted options={
    fontsize=\scriptsize,
    breaklines=true
  },
  listing above comment,
  middle=0pt,
  comment={
    \begin{commentb}
      #2
    \end{commentb}
  },
  #1,
}
