import re
import os
import base64

def image_to_base64(path):
    # Try absolute path first or relative
    if not os.path.exists(path):
        # Try resolving relative to current working directory
        possible_path = os.path.join(os.getcwd(), path)
        if not os.path.exists(possible_path):
            print(f"File not found: {path} or {possible_path}")
            return None
        path = possible_path
    
    # Detect mime type
    mime = "image/png"
    if path.endswith(".jpg") or path.endswith(".jpeg"):
        mime = "image/jpeg"
    elif path.endswith(".gif"):
        mime = "image/gif"
        
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime};base64,{encoded_string}"

def main():
    md_file = "relatório-lifelines.md"
    if not os.path.exists(md_file):
        print(f"{md_file} not found.")
        return
        
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Match markdown images: ![alt](path)
    pattern = r'!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif))\)'
    
    def replacer(match):
        alt = match.group(1)
        path = match.group(2)
        # Clean path
        path_clean = path.strip().replace('\\', '/')
        b64 = image_to_base64(path_clean)
        if b64:
            print(f"Successfully embedded: {path_clean}")
            return f"![{alt}]({b64})"
        return match.group(0)
        
    new_content = re.sub(pattern, replacer, content)
    
    out_file = "relatório-lifelines-embedded.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Saved base64 embedded markdown to: {out_file}")

if __name__ == "__main__":
    main()
