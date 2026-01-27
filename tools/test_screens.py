#!/usr/bin/env python3
"""Interactive tester for individual screen modules."""
import ast
import importlib.util
import inspect
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENS_DIR = PROJECT_ROOT / "screens"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SCREEN_DELAY
import utils


def load_display():
    main_path = PROJECT_ROOT / "main.py"
    namespace = {"__file__": str(main_path)}
    src = main_path.read_text(encoding="utf-8")
    guard = src.find("if __name__")
    if guard != -1:
        src = src[:guard]
    exec(compile(src, str(main_path), "exec"), namespace)
    return namespace.get("display"), namespace

def list_screens(script_dir, namespace):
    screens = []
    # scroll logos
    screens.append({"name":"scroll_all_logos","type":"scroll"})
    for fname in sorted(os.listdir(script_dir)):
        if not (fname.endswith(".py") and (fname.startswith("draw_") or fname == "mlb_scoreboard.py")):
            continue
        mod = os.path.splitext(fname)[0]
        path= os.path.join(script_dir,fname)
        try:
            tree=ast.parse(open(path).read())
        except:
            continue
        fns=[]
        for node in tree.body:
            if isinstance(node,ast.FunctionDef):
                low=node.name.lower()
                if ("draw" in low or "show" in low) and "get" not in low:
                    fns.append(node.name)
        if fns:
            screens.append({"name":f"{mod}.py","type":"module","module":mod,"funcs":fns})
            for fn in fns:
                screens.append({"name":f"{mod}.{fn}","type":"function","module":mod,"func":fn})
    return screens

def invoke_screen(entry, display, namespace):
    if entry["type"]=="scroll":
        for var in [
            "weather_img",
            "verano_img",
            "hawks_logo",
            "wolves_logo",
            "bulls_logo",
            "cubs_logo",
            "sox_logo",
            "mlb_logo",
            "bears_logo",
        ]:
            img = namespace.get(var)
            if img:
                utils.animate_scroll(display, img)
                time.sleep(SCREEN_DELAY/2)
                utils.clear_display(display)
        return

    script_dir=str(SCREENS_DIR)
    fname=entry["module"]+".py"
    spec=importlib.util.spec_from_file_location(entry["module"],os.path.join(script_dir,fname))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fns=[]
    if entry["type"]=="module":
        for name in entry["funcs"]:
            fns.append(getattr(module,name))
    else:
        fns=[getattr(module,entry["func"])]
    for fn in fns:
        sig=inspect.signature(fn)
        params=list(sig.parameters.keys())
        args=[display]
        for p in params[1:]:
            if hasattr(module,p):
                args.append(getattr(module,p))
            elif hasattr(module,f"get_{p}"):
                args.append(getattr(module,f"get_{p}")())
            else:
                args.append(None)
        result = fn(*args)
        already_displayed = False
        img = result
        if isinstance(result, utils.ScreenImage):
            img = result.image
            already_displayed = result.displayed
        if img and not already_displayed:
            display.image(img)
        display.show()
        time.sleep(SCREEN_DELAY)
        utils.clear_display(display)

def main():
    script_dir = str(SCREENS_DIR)
    display, namespace = load_display()
    while True:
        screens=list_screens(script_dir,namespace)
        if not screens:
            print("No screens found."); sys.exit(1)
        print("\nAvailable screens:")
        for idx,e in enumerate(screens,1):
            name=e["name"]
            print(f"{idx}. {name}")
        choice=input("\nEnter numbers (e.g. 1,3-5) or 'q' to quit: ")
        if choice.strip().lower()=="q":
            break
        sel=set()
        for tok in re.split(r"\s*,\s*",choice):
            if "-" in tok:
                try: a,b=map(int,tok.split("-")); sel.update(range(a-1,b))
                except: pass
            else:
                try: sel.add(int(tok)-1)
                except: pass
        valid=[i for i in sel if 0<=i<len(screens)]
        if not valid:
            print("No valid selections."); continue
        for i in sorted(valid):
            print(f"\nShowing: {screens[i]['name']}")
            invoke_screen(screens[i],display,namespace)
    if input("\nStart service? (y/n): ").lower().startswith("y"):
        os.system("sudo systemctl start display.service")
        print("Service started.")
    print("Goodbye.")

if __name__=="__main__":
    main()
