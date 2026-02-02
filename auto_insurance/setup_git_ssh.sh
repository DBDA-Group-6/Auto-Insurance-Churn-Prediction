#!/bin/bash
# ==============================================================================
#  setup_git_ssh.sh
#  Connects your EC2 Ubuntu instance to your GitHub account over SSH.
#
#  What it does:
#    1. Generates an SSH key-pair (~/.ssh/id_ed25519)  — skips if one exists
#    2. Prints your public key  → you paste it into GitHub Settings → SSH Keys
#    3. Tests the SSH connection to github.com
#    4. Configures your Git user.name & user.email globally
#    5. Clones your repository into ~/project
#
#  Usage:
#    chmod +x setup_git_ssh.sh
#    sudo ./setup_git_ssh.sh
# ==============================================================================

set -e

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   EC2 ↔ GitHub SSH Setup Script${NC}"
echo -e "${BLUE}============================================================${NC}"

# ─── 1. Determine the real (non-root) user ───────────────────────────────────
if [ -n "$SUDO_USER" ]; then
    REAL_USER="$SUDO_USER"
else
    REAL_USER="$USER"
fi
HOME_DIR=$(eval echo "~${REAL_USER}")
SSH_DIR="${HOME_DIR}/.ssh"
KEY_PATH="${SSH_DIR}/id_ed25519"

echo -e "\n${GREEN}[INFO]${NC} Running as root, operating on user: ${YELLOW}${REAL_USER}${NC}"
echo -e "${GREEN}[INFO]${NC} SSH directory : ${YELLOW}${SSH_DIR}${NC}"
echo -e "${GREEN}[INFO]${NC} Key path      : ${YELLOW}${KEY_PATH}${NC}"

# ─── 2. Ensure .ssh directory exists with correct permissions ─────────────────
mkdir -p "${SSH_DIR}"
chmod 700 "${SSH_DIR}"
chown "${REAL_USER}:${REAL_USER}" "${SSH_DIR}"

# ─── 3. Generate SSH key (ed25519 — modern, short, secure) ───────────────────
if [ -f "${KEY_PATH}" ]; then
    echo -e "\n${YELLOW}[SKIP]${NC} SSH key already exists at ${KEY_PATH}"
else
    echo -e "\n${GREEN}[STEP 1]${NC} Generating SSH key pair …"
    ssh-keygen -t ed25519 -f "${KEY_PATH}" -N "" -C "${REAL_USER}@ec2-github"
    chown "${REAL_USER}:${REAL_USER}" "${KEY_PATH}" "${KEY_PATH}.pub"
    echo -e "${GREEN}[DONE]${NC}  Key pair created."
fi

# ─── 4. Print public key ─────────────────────────────────────────────────────
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}   YOUR PUBLIC KEY  — Copy everything between the dashes     ${NC}"
echo -e "${BLUE}============================================================${NC}"
cat "${KEY_PATH}.pub"
echo -e "${BLUE}============================================================${NC}"

echo -e "\n${YELLOW}[ACTION REQUIRED]${NC}"
echo -e "  1. Copy the public key shown above."
echo -e "  2. Open a browser → go to:  https://github.com/settings/keys/new"
echo -e "  3. Give it a title, e.g.  \"EC2 Ubuntu\""
echo -e "  4. Paste the key into the \"Key\" text box."
echo -e "  5. Click  \"Add SSH key\"."
echo -e "\n  Press ${GREEN}ENTER${NC} when you have added the key to GitHub …"
read -r

# ─── 5. Test SSH connection to GitHub ────────────────────────────────────────
echo -e "\n${GREEN}[STEP 2]${NC} Testing SSH connection to github.com …"
# Run as the real user so it picks up the correct key
if sudo -u "${REAL_USER}" ssh -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | grep -q "Hi "; then
    GITHUB_USER=$(sudo -u "${REAL_USER}" ssh -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | grep -oP 'Hi \K[^!]+')
    echo -e "${GREEN}[DONE]${NC}  Connected successfully!  GitHub user: ${YELLOW}${GITHUB_USER}${NC}"
else
    echo -e "${RED}[ERROR]${NC} SSH connection failed."
    echo -e "  • Make sure you added the key on GitHub before pressing ENTER."
    echo -e "  • You can retry by running this script again."
    exit 1
fi

# ─── 6. Configure Git identity ───────────────────────────────────────────────
echo -e "\n${GREEN}[STEP 3]${NC} Configuring Git identity …"

GIT_NAME="DBDA-Group-6"

GIT_EMAIL="zukiroina@gmail.com"

#echo -e "  Enter your Git user name  (e.g. John Doe): "
#read -r GIT_NAME

#echo -e "  Enter your Git email      (e.g. you@email.com): "
#read -r GIT_EMAIL

sudo -u "${REAL_USER}" git config --global user.name  "${GIT_NAME}"
sudo -u "${REAL_USER}" git config --global user.email "${GIT_EMAIL}"

echo -e "${GREEN}[DONE]${NC}  Git identity set:"
echo -e "        name  → ${YELLOW}${GIT_NAME}${NC}"
echo -e "        email → ${YELLOW}${GIT_EMAIL}${NC}"

# ─── 7. Clone repository ─────────────────────────────────────────────────────

REPO_URL="git@github.com:DBDA-Group-6/Auto-Insurance-Churn-Prediction.git"

#echo -e "\n${GREEN}[STEP 4]${NC} Clone your repository …"
#echo -e "  Enter your GitHub repo SSH URL"
#echo -e "  (e.g. git@github.com:yourusername/yourrepo.git): "
#read -r REPO_URL

CLONE_DIR="${HOME_DIR}/project"

if [ -d "${CLONE_DIR}" ]; then
    echo -e "\n${YELLOW}[WARN]${NC} ${CLONE_DIR} already exists."
    echo -e "  Choose an option:"
    echo -e "    1) Remove it and clone fresh"
    echo -e "    2) Clone into a different folder (${HOME_DIR}/project2)"
    echo -e "    3) Skip cloning"
    echo -e "  Your choice [1/2/3]: "
    read -r CHOICE
    case "$CHOICE" in
        1) rm -rf "${CLONE_DIR}" ;;
        2) CLONE_DIR="${HOME_DIR}/project2" ;;
        3) echo -e "${YELLOW}[SKIP]${NC} Cloning skipped."; CLONE_DIR="" ;;
    esac
fi

if [ -n "${CLONE_DIR}" ]; then
    sudo -u "${REAL_USER}" git clone "${REPO_URL}" "${CLONE_DIR}"
    echo -e "${GREEN}[DONE]${NC}  Repository cloned to  ${YELLOW}${CLONE_DIR}${NC}"
fi

# ─── 8. Summary ──────────────────────────────────────────────────────────────
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}   Setup Complete — Quick Reference${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "  SSH Key      :  ${KEY_PATH}"
echo -e "  Git user     :  ${GIT_NAME} <${GIT_EMAIL}>"
echo -e "  GitHub user  :  ${GITHUB_USER}"
if [ -n "${CLONE_DIR}" ]; then
    echo -e "  Project dir  :  ${CLONE_DIR}"
    echo ""
    echo -e "  Common commands:"
    echo -e "    cd ${CLONE_DIR}"
    echo -e "    git pull                   # pull latest changes"
    echo -e "    git add .                  # stage all changes"
    echo -e "    git commit -m \"message\"    # commit"
    echo -e "    git push origin main       # push to GitHub"
fi
echo -e "${BLUE}============================================================${NC}"
