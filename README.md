# HandsOnAutonomousAerialVehicles

## How to add and push a specific folder to this repository

If you want to add and push only a particular folder (e.g., `pkatyal_p0`) to the repository, use the following commands:

```bash
cd ~/HoAAV
git add #Folder name
git commit -m "Add pkatyal_p0 folder"
git push origin main
```

This will stage, commit, and push only the changes in the `pkatyal_p0` folder to the remote repository.
## How to activate the HoAAV Python virtual environment



To activate the environment:
```bash
source ~/HoAAV/HoAAV/bin/activate
```

After activation, your shell prompt should show `(HoAAV)` at the beginning.
