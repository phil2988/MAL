from tkinter import (
    Tk,
    _setit,
    HORIZONTAL,
    Frame,
    BOTH,
    Button,
    Entry,
    Label,
    Text,
    scrolledtext,
    END,
    filedialog,
    Menu,
    Radiobutton,
    IntVar,
    ttk,
    OptionMenu,
    StringVar,
    Toplevel,
    Checkbutton,
    Canvas,
)
import os
import sys
import traceback
from preprocessing import getCardsAsDataFrameByPath


class APP(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.set_class_vars()
        self.setWindowPropperties()
        self.create_wigtes()

    def setWindowPropperties(self):
        self.width = 400
        self.height = 600
        self.master.geometry(f"{self.width}x{self.height}")
        self.master.resizable(0, 0)
        self.master.title("Phillip Labeling tool")
        try:
            self.master.iconbitmap(default=makeEXEPath("icon.ico"))
        except Exception:
            print(traceback.format_exc())

    def set_class_vars(self):
        self.file_in_path = None
        self.out_path = None
        self.df = None
        self.index = 0

    def build_data(self, df):
        pass

    def next(self):
        if self.index + 1 < self.df.shape[0]:
            self.index += 1
            self.fill_in_card()
        else:
            # All has been labled
            print("Done")

    def previous(self):
        self.index -= 1
        if self.index - 1 < self.df.shape[0]:
            self.fill_in_card()
        else:
            # All has been labled
            print("Done")

    def fill_in_card(self):
        if str(self.index) + "_label.txt" in os.listdir(self.out_path):
            with open(
                os.path.join(self.out_path, str(self.index) + "_label.txt"), "r"
            ) as f:
                label = f.read().strip().replace("\n", "")

            self.current_label.configure(text=f"Current label: {label}")
        else:
            self.current_label.configure(text="Current label: None")

        # Cleanup
        for widget in self.frame_card.winfo_children():
            widget.destroy()

        col1_y_offset = 0
        col2_y_offset = 0

        for i, col in enumerate(self.df):
            x = int(10 if i <= 10 else 200)

            y = int((i if i <= 10 else (i - 11)) * (300 / (self.df.shape[1] / 2)) + 10)

            y += col1_y_offset if i <= 10 else col2_y_offset

            txt = f"{col}: - {self.df.iloc[self.index][col]}".strip()

            for _i in range(0, len(txt), 29):
                if _i == 0:
                    continue
                txt = txt[0:_i] + "\n" + txt[_i : len(txt) - 1]

                if i <= 10:
                    col1_y_offset += 15
                else:
                    col2_y_offset += 15

            Label(self.frame_card, text=txt).place(x=x, y=y)

    def create_wigtes(self):
        # Chose in-file
        Button(self.master, text="...", command=self.ask_file, width=3).place(
            x=15, y=10
        )
        self.label_in_file_path = Label(text="Choose data source")
        self.label_in_file_path.place(x=55, y=13)

        # Chose out-path
        Button(self.master, text="...", command=self.ask_directory, width=3).place(
            x=15, y=40
        )
        self.label_out_path = Label(text="Choose out dir")
        self.label_out_path.place(x=55, y=43)

        # Card
        self.frame_card = Frame(self.master, width=370, height=420, background="white")
        self.frame_card.place(x=15, y=75)

        # Label
        self.current_label = Label(
            self.master, text="Current label: None", font=("Arial", 11)
        )
        self.current_label.place(x=15, y=503)

        # buttons
        self.btn_prev = Button(
            self.master, text="Prev", command=self.previous, state="disabled"
        )
        self.btn_prev.place(x=15, y=530, width=370 / 2 - 2.5)
        self.btn_next = Button(
            self.master, text="Next", command=self.next, state="disabled"
        )
        self.btn_next.place(x=15 + 370 / 2, y=530, width=370 / 2 - 2.5)

        self.buttons = [
            Button(
                self.master,
                text="aggro",
                command=lambda: self.apply_label("aggro"),
                state="disabled",
            ),
            Button(
                self.master,
                text="control",
                command=lambda: self.apply_label("control"),
                state="disabled",
            ),
            Button(
                self.master,
                text="tempo",
                command=lambda: self.apply_label("tempo"),
                state="disabled",
            ),
        ]

        for i, b in enumerate(self.buttons):
            btn_space_px = (self.width - 15 - 15) / (len(self.buttons))
            b.place(x=10 + (i * btn_space_px + 5), y=560, width=btn_space_px)

    def check_for_enable(self):
        if all([self.out_path is not None, self.file_in_path is not None]):
            [
                b.configure(state="normal")
                for b in self.buttons + [self.btn_next, self.btn_prev]
            ]

            self.fill_in_card()

    # Gets the rootpath
    def ask_directory(self, event=None):
        temp_path = filedialog.askdirectory()

        if not os.path.isdir(temp_path):
            # Validate path
            self.label_out_path.configure(text="Invalid path")
            self.out_path = None
            return

        self.out_path = temp_path

        # Adabt the path to the textfield
        filename = "..." + temp_path[-41:] if len(temp_path) >= 39 else temp_path

        self.label_out_path.configure(text=filename)

        self.check_for_enable()

    def ask_file(self, event=None):
        temp_path = filedialog.askopenfilename()

        if not os.path.isfile(temp_path) or not self.load_data(temp_path):
            # Validate path
            self.label_in_file_path.configure(text="Invalid path")
            self.file_in_path = None
            return

        self.file_in_path = temp_path

        # Adabt the path to the textfield
        filename = "..." + temp_path[-41:] if len(temp_path) >= 39 else temp_path

        self.label_in_file_path.configure(text=filename)

        self.check_for_enable()

    def apply_label(self, label: str):
        self.df.iloc[self.index].to_csv(
            os.path.join(self.out_path, f"{self.index}.csv"), index=False, header=False
        )
        with open(
            os.path.join(self.out_path, str(self.index) + "_label.txt"), "w+"
        ) as f:
            f.write(label)
        self.next()

    def load_data(self, path):
        try:
            self.df = getCardsAsDataFrameByPath(path)
            return True
        except Exception:
            print(traceback.format_exc())
            return False
            # Error here


def makeEXEPath(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


if __name__ == "__main__":
    root = Tk()
    app = APP(root)
    try:
        root.mainloop()
    except Exception:
        pass
    finally:
        print("Cleanup")
        # Cleanup
