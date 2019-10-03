
def insert_img(scale, img_address, caption):
    print("\\begin{figure}[H]\n	\centering \n\
    \includegraphics[scale=", str(scale), "]",img_address,"\n\
    \caption{", caption, "}\n\end{figure}")

def insert_img2(scale, img_address1, img_address2,
                caption1, caption2):
    print("\\begin{figure}[H]\n	\centering \n\
    \\begin{minipage}[h]{0.3\\textwidth}\n \
        \includegraphics[scale=", str(scale), "]", img_address1, "\n\
        \caption{", caption1, "}\n\
    \end{minipage}\n\
    \hfill\n\
    \\begin{minipage}[h]{0.3\\textwidth}\n \
        \includegraphics[scale=", str(scale), "]", img_address2, "\n\
        \caption{", caption2, "}\n\
    \end{minipage}\n\
\end{figure}")

def insert_img3(scale, img_address1, img_address2,
                img_address3,
                caption1, caption2, caption3):
    print("\\begin{figure}[H]\n	\centering \n\
    \\begin{minipage}[h]{0.3\\textwidth}\n \
        \includegraphics[scale=", str(scale), "]", img_address1, "\n\
        \caption{", caption1, "}\n\
    \end{minipage}\n\
    \hfill\n\
    \\begin{minipage}[h]{0.3\\textwidth}\n \
        \includegraphics[scale=", str(scale), "]", img_address2, "\n\
        \caption{", caption2, "}\n\
    \end{minipage}\n\
    \hfill\n\
    \\begin{minipage}[h]{0.3\\textwidth}\n \
        \includegraphics[scale=", str(scale), "]", img_address3, "\n\
        \caption{", caption3, "}\n\
    \end{minipage}\n\
\end{figure}")


def img_table(height, width, img_add, caption):
    print("\\begin{figure}[H]\n	\centering \n")

    for h in range(height):
        for w in range(width):
            print("    \\begin{subfigure}[t]{\\textwidth} \n \
        \\centering \n \
        \\includegraphics[width=\\linewidth]", img_add[w + width * h], " \n \
    \\end{subfigure}")
            if w < width - 1:
                print("\\hfill")
        if h < height - 1:
            print("\\vspace{1cm}")

    print("    \\caption{", caption, "} \n "
                                     "\\end{figure}")


def insert_img3_onecap(scale, img_address1, img_address2,
                img_address3, caption):
    print("\\begin{figure}[H]\n	\centering \n\
    \\begin{minipage}[h]{0.3\\textwidth}\n \
        \includegraphics[scale=", str(scale), "]", img_address1, "\n\
    \end{minipage}\n\
    \hfill\n\
    \\begin{minipage}[h]{0.3\\textwidth}\n \
        \includegraphics[scale=", str(scale), "]", img_address2, "\n\
    \end{minipage}\n\
    \hfill\n\
    \\begin{minipage}[h]{0.3\\textwidth}\n \
        \includegraphics[scale=", str(scale), "]", img_address3, "\n\
    \end{minipage}\n\
    \caption{", caption, "}\n \
\end{figure}")

def img8_input(scale, img_add, caption):
    # this function get caption and address of images and return command
    # for table of 2*4 of images in latex
    print("\\begin{table}\n\
	\\centering\n\
	\\begin{tabular}{m{35mm}m{35mm}m{35mm}m{35mm}}\n\
		\includegraphics[scale=", str(scale), "  ]", img_add[0] ,"&\n\
		\includegraphics[scale=", str(scale), "  ]", img_add[1] ,"&\n\
		\includegraphics[scale=", str(scale), "  ]", img_add[2] ,"&\n\
		\includegraphics[scale=", str(scale), "  ]", img_add[3] ,"\\\\ \n\
		\includegraphics[scale=", str(scale), "  ]", img_add[4], "&\n\
        \includegraphics[scale=", str(scale), "  ]", img_add[5], "&\n\
        \includegraphics[scale=", str(scale), "  ]", img_add[6], "&\n\
        \includegraphics[scale=", str(scale), "  ]", img_add[7], " \n\
	\end{tabular}\n\
	\caption{", caption, "}\n\
\end{table}")







