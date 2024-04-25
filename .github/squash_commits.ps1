param(
    [string]$CommitMessage,
    [string]$fork
)

$mainBranch = 'main'
$featureBranch = git rev-parse --abbrev-ref HEAD

git fetch origin $mainBranch

# Check out the feature branch (though you should already be on it)
git checkout $featureBranch

git rebase -i origin/$mainBranch

# Squash commits
# Note: The interactive rebase '-i' will open an editor to squash commits manually
# Replace 'pick' with 'squash' beside all but the first commit to combine them

# If you're not comfortable with the interactive mode or want to automate:
# Assuming you want to squash all commits made on the feature branch since it diverged from main:
$commitCount = (git rev-list --count HEAD ^origin/$mainBranch)
if ($commitCount -gt 1) {
    git reset --soft "HEAD~$commitCount"
    git commit -m $CommitMessage
}

# Push changes to the remote repository
git push $fork $featureBranch --force
