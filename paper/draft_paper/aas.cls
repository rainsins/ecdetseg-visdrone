\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{aas}[2026/09/09 v1.0 AAS CJK LaTeX template class by latexstudio-xmy] % Need to be changed with new release
%
\LoadClass[a4paper, zihao = 5, twoside, twocolumn, fontset = fandol, final]{ctexart}
\RequirePackage{ifthen}
\RequirePackage{
  amsmath, amssymb, amsthm, amsfonts, graphicx, hyperref, flushend, picins,
  calc, iftex, enumitem, multicol, subfigure, ccaption, booktabs, float, array,
  newtxmath}
\let \bicaption \relax
\RequirePackage{bicaption}
\captionsetup[figure][bi-first]
  {name = 图,   justification = centering, labelsep = quad, font = small}
\captionsetup[figure][bi-second]
  {name = Fig., justification = centering, labelsep = quad, font = small}
\captionsetup[table][bi-first]
  {name = 表,   justification = centering, labelsep = quad, font = small}
\captionsetup[table][bi-second]
  {name = Table,justification = centering, labelsep = quad, font = small}
\ifXeTeX \else \RequirePackage{epstopdf} \fi
\hypersetup{hidelinks}
\newcommand \mailto[1] {\href{mailto:#1}{E-mail:~\texttt{#1}}}
%

\newtheoremstyle{aastheorem}%
  {2pt}{2pt}%
  {}{2\ccwd}%
  {\bfseries}{.}%
  {1em}%
  {\thmname{\heiti #1} \thmnumber{#2}\thmnote{ (#3)}}

\theoremstyle{aastheorem}

\newtheorem{theorem}{定理}
\newtheorem{example}{例}
\newtheorem{definition}{定义}
\newtheorem{corollary}{推论}
\newtheorem{lemma}{引理}

%
%%--------PAGE FORMAT---------%%
\setlength \topmargin     {0mm}
\setlength \headsep       {15pt}    % 页眉底 与正文上面之间的距离
\setlength \headheight    {6mm}
\setlength \oddsidemargin {-7mm}
\setlength \evensidemargin{-7mm}
\setlength \voffset       {-0.40in} % one inch + \voffset 页眉顶到页顶的距离
\setlength \hoffset       {0mm}
\setlength \footskip      {3.5em}
\setlength \columnsep     {6mm}     % 设置两栏之间的间距

\textwidth \paperwidth%
\textheight \paperheight%
\addtolength\textwidth{-38mm} 
\addtolength\textheight{-62mm} 
% extra vertical space between paragraphs normally
\setlength{\parskip}{0mm}
% extra vertical space between paragraphs within a list item
%\setlength{\parsep}{0mm}

%% 下面两条命令设定displaymath与上下文的间距
\setlength{\abovedisplayskip}{8pt plus 2pt minus 2.5pt}
\setlength{\belowdisplayskip}{8pt plus 2pt minus 2.5pt}

\setlength{\doublerulesep}{0.5pt}

%% 行距
\renewcommand{\baselinestretch}{1.15} % 1.22 --- 1.25倍行距
\renewcommand{\arraystretch}{1.16}
%
\newlength\aas@linespace
\newcommand{\aas@choosefont}[3]{%
   \setlength{\aas@linespace}{#1*\real{\baselinestretch}}%
   \fontsize{#1}{\aas@linespace plus#2 minus #3}\selectfont}

\normalsize % make \baselinestretch take affect
% 改文献的序号[1]为 1
\renewcommand{\@biblabel}[1]{#1}
% 使整篇文章的首段都有缩进
\let\@afterindentfalse\@afterindenttrue\@afterindenttrue
% 参考文献号上标显示
\def\@cite#1#2{\textsuperscript{[{#1\if@tempswa, #2\fi}]}}
%
% 两脚注间的竖直距离
\setlength{\footnotesep}{0pt}
% 重新定义 脚注条
\def\footnoterule{\kern-3\p@ \hrule width .7\columnwidth height 0.4pt \kern 2.6\p@}
%
%%-------- maketitle --------%%
%
\newcommand{\cntitle}[1]{\gdef\aas@cntitle{{\zihao{-2}\heiti #1}}}
\newcommand{\cnauthor}[1]{\gdef\aas@cnauthor{\zihao{-4} \fangsong #1}}
\newcommand{\doi}[1]{%
  \gdef\aas@doi{%
    {\noindent\small \textbf{DOI}\quad \href{https://doi.org/#1}{#1}}}}
\newcommand{\cstr}[1]{%
  \gdef\aas@cstr{%
    {\noindent\small \textbf{CSTR}\quad #1}}}
\newcommand{\entitle}[1]{\gdef\aas@entitle{\textbf{\large \sffamily #1}}}
\newcommand{\enauthor}[1]{\gdef\aas@enauthor{\small #1}}
\newcommand{\enaddress}[1]{\gdef\aas@enaddress{{\noindent\small \textbf{Citation}\quad #1}}}
\newcommand{\email}[1]{\gdef\aas@email{{\centering\footnotesize (E-mail: #1)}}}
\newcommand{\enabstract}[1]{\gdef\aas@enabstract{{\noindent\small \textbf{Abstract}\quad #1}}}
\newcommand{\enkeyword}[1]{\gdef\aas@enkeyword{{\noindent\small \textbf{Keywords}\quad #1}}}
%
\renewcommand{\refname}{\centering \normalsize\textbf{References}}
%
\renewcommand{\maketitle}{\par
 \begingroup
  \def\thefootnote{}%  the \thanks{} mark type is empty
  \def\footnotemark{}% and kill space from \thanks within author
   \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
   \long\def\@makefntext##1{\noindent
      \qquad \hb@xt@\z@{%
               \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
   \twocolumn[\@maketitle]%
   \thispagestyle{aas@firstheadings}%ssl
   \@thanks%
   \vspace{-1.75\baselineskip}
 \endgroup%
 \setcounter{footnote}{0}%
 \global\let\thanks\relax%
 \global\let\maketitle\relax%
 \global\let\@maketitle\relax%
 \global\let\@thanks\@empty%
 \global\let\@author\@empty%
 \global\let\@title\@empty%
 \global\let\title\relax%
 \global\let\author\relax%
 \global\let\and\relax%
%
 \global\let\aas@cntitle\relax%
 \global\let\aas@cnauthor\relax%
 \global\let\aas@cnaddress\relax%
 \global\let\aas@cnabstract\relax%
 \global\let\aas@cnkeyword\relax%
 \global\let\aas@doi\relax%
 \global\let\aas@cstr\relax%
%
 \global\let\cntitle\relax%
 \global\let\cnauthor\relax%
 \global\let\cnaddress\relax%
 \global\let\cnabstract\relax%
 \global\let\cnkeyword\relax%
 \global\let\cncl\relax%
 \global\let\email\relax%
 \global\let\doi\relax%
%
 \global\let\aas@entitle\relax%
 \global\let\aas@enauthor\relax%
 \global\let\aas@enaddress\relax%
 \global\let\aas@enabstract\relax%
 \global\let\aas@enkeyword\relax%
%
 \global\let\entitle\relax%
 \global\let\enauthor\relax%
 \global\let\enaddress\relax%
 \global\let\enabstract\relax%
 \global\let\enkeyword\relax%
}
%
\def\@maketitle{%
    \let\footnote\thanks%
    \newpage\null% an empty hbox
    \vskip 1.0\baselineskip%
    {\centering%
    {\aas@cntitle\par}%
    {%\vskip 0.8em%
    \vskip 0.8\baselineskip%
     \begin{tabular}[t]{c}%
     \aas@cnauthor%
     \end{tabular}%
     \vskip 0.3em%
     \par%
     }%
     \vskip 0.2\baselineskip}%
    {\linespread{1.0}\selectfont\cabstract\par%
    \vskip 0.2\baselineskip%
    \ckeyword\par%
    \vskip 0.2\baselineskip%
    \caddress\par%  
    \vskip 0.2\baselineskip% 
    \aas@doi
    \qquad
    \aas@cstr
    \par
    }%
    \vskip 0.7\baselineskip%
    {\centering%
    {\aas@entitle\par}%
    {%\vskip 1.2em%
    \vskip 0.5\baselineskip%
    \begin{tabular}[t]{c}%
    \aas@enauthor%
    \end{tabular}%
    \vskip 0.3em%
    \par%
    }%
    \vskip 0.2\baselineskip}%
    {\linespread{1.0}\selectfont\aas@enabstract\par%
    \vskip 0.2\baselineskip%
    \aas@enkeyword\par%
    \vskip 0.2\baselineskip%
    \aas@enaddress\par%    
    }%
    \vskip 1.5\baselineskip%
}
%%---------------- Define biography ----------------%%
\newenvironment{biography}[1][]{
\vskip 0.4\baselineskip%
\zihao{-5}
\parpic{%
\includegraphics[height=3.0cm,width=2.4cm, keepaspectratio]{#1}}%
}{\vskip 0.4\baselineskip\par}


%% For author who has no photo
\newenvironment{biographynophoto}{\vskip 0.4\baselineskip\zihao{-5}}{\vskip 0.4\baselineskip\par}

%%---------------- Redefine section -----------------%%
\setcounter{secnumdepth}{3}

%%-------- Set the begin and end method for class -----%%
\AtBeginDocument{%
   \flushend

\newcommand*{\cnabstract}[1]{\gdef\cabstract{\noindent{\zihao{-5}{\heiti 摘\quad 要}\quad #1}}}
\newcommand*{\cnkeyword}[1]{\gdef\ckeyword{\noindent{\zihao{-5}{\heiti 关键词}\quad #1}}}
\newcommand*{\cnaddress}[1]{\gdef\caddress{\noindent{\zihao{-5}{\heiti 引用格式}\quad #1}}}
%
%%------------------aas@firstheadings begin------------------------------------%%
\def\ps@aas@firstheadings{
   \let\@evenhead\@empty\let\@evenfoot\@empty
   \let\@oddhead\@empty\let\@oddfoot\@empty
   \def\@evenhead{%
       \vbox{%
          \hbox to \textwidth{\zihao{-5}\songti\space
            第 XX 卷\hskip 0.8em第 X 期\hfill
            自\hskip 0.79em动\hskip 0.79em化\hskip 0.79em学\hskip 0.79em报\hfill Vol.\,XX, No.\,X\space}%
          \vskip 2.13mm%
          \hbox to \textwidth{\zihao{-5}\songti\space
            202X 年 X 月\hskip8mm\hfill
            ACTA AUTOMATICA SINICA\hfill Month 202X\space}%
          \vskip 1.35mm%
          \hrule width\textwidth height0.63pt\vskip0.85pt \hrule width\textwidth height0.63pt%
       }
   }
   %\def\@evenfoot{
   %\hbox{}%\zihao{-5}$\copyright$ 200X by {\slshape Acta Automatica Sinica}. All rights reserved.
   %}
   \let\@oddhead\@evenhead%
   \let\@oddfoot\@evenfoot%
}

%%-----------------aas@firstheadings end--------------------------------------%%

%%-------------------aasheadings begin----------------------------------------%%
\def\ps@aasheadings{
    \let\@evenhead\@empty\let\@evenfoot\@empty
    \let\@oddhead\@empty\let\@oddfoot\@empty
    \def\@evenhead{%
        \vbox{%
            \vskip 3.2mm%
            \hbox to \textwidth {\zihao{-5}\songti\space
                \hbox{\thepage} \hfill
                \hbox{自\hskip 2.30em动\hskip 2.30em化\hskip 2.30em学\hskip 2.30em报}\hfill
                \hbox{XX 卷}\space
            }%
            \vskip 2.0mm%
            \hbox to \textwidth{\noindent\rule[2mm]{\textwidth}{0.5pt}}%
        }%
    }
    \def\@oddhead{%
        \vbox{%
            \vskip 3.2mm%
            \hbox to \textwidth {\space {\zihao{-5} X\hskip 0.18em}{\zihao{-5}\songti 期}\hfill{\small\zihao{-5}\songti 尚书林等: 《自动化学报》稿件加工样本}\hfill {\zihao{-5}\thepage\space}}%
            \vskip 2.0mm%
            \hbox to \textwidth{\noindent\rule[2mm]{\textwidth}{0.5pt}}%  
        }
    }%
}
%---------------------headings end------------------------------------------%%


   \captiondelim{ } % 把图标题中缺省的分隔符 ``:''改为空
   \captionnamefont{\small}%
   \captiontitlefont{\small}%
   }
\AtEndDocument{\clearpage}

\ctexset{%
  space = false,
  bibname = \makebox[\linewidth]{\zihao{5}参考文献},
  figurename = 图,
  tablename  = 表,
  section = {
    fixskip,
    beforeskip = 1.5ex,
    afterskip  = 1.5ex,
    format+ = \raggedright \zihao{-4}\sffamily\mdseries
  },
  subsection = {
    fixskip,
    beforeskip = 1.0ex,
    afterskip  = 1.0ex,
    format+ = \raggedright \zihao{5}\sffamily\mdseries
  },
  subsubsection = {
    fixskip,
    beforeskip = .5ex,
    afterskip  = .5ex,
    format+ = \raggedright \zihao{5}\sffamily\mdseries
  }
}
\setlist{noitemsep, topsep = 1pt}


\RequirePackage{fancyvrb}

\def\VerbLB{\FV@Command{}{VerbLB}}
\begingroup
\catcode`\^^M=\active%
\gdef\FVC@VerbLB#1{%
  \begingroup%
    \FV@UseKeyValues%
    \FV@FormattingPrep%
    \FV@CatCodes%
    \def^^M{ }%
    \catcode`#1=12%
    \def\@tempa{\def\FancyVerbGetVerb####1####2}%
    \expandafter\@tempa\string#1{\mbox{##2}\endgroup}%
    \FancyVerbGetVerb\FV@EOL}%
\endgroup

\endinput
%%-------- The end of aas.cls -------------%%
