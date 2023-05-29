# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


from pwem.viewers import showj
from pwem.viewers.showj import runJavaIJapp
from pyworkflow.gui import *


def launchIJForSelection(path, image):
    macroPath = os.path.join(path, "AutoSave_ROI.ijm")
    imageBaseName = os.path.basename(image)

    macro = r"""
    path = "%s";
    file = "%s"
    
    // --------- Initialize Roi Manager ---------
    roiManager("Draw");
    setTool("freehand");
    run("Fire");
    run("Scale to Fit");
    
    newClass = "Yes";
    outPath = path + file + ".txt";
    
    // --------- Load SetOfMeshes ---------
    if (File.exists(outPath)){
    group = 0;
    groups = loadMeshFile(outPath);
    numMeshes = roiManager("count");
    emptyOutFile(outPath);
    aux = editMeshes(groups, numMeshes, outPath);
    group = group + aux + 1;
    }
    else{
    emptyOutFile(outPath);
    group = 1;
    }
    
    // --------- Draw new Meshes and save them ---------
    roiManager("Reset");
    waitForRoi();
    saveMeshes(group, outPath);
    
    // --------- Close ImageJ ---------
    run("Quit");
    
    
    // --------- Functions Definition ---------        
    function waitForRoi(){
    waitForUser("Draw the desired ROIs\n\nThen click Ok");
    wait(50);
    while(roiManager("count")==0){
    waitForUser("Draw the desired ROIs\n\nThen click Ok");
    wait(50);
    }
    }
    
    function emptyOutFile(outPath){
    fid = File.open(outPath);
    File.close(fid);
    }
    
    function saveMeshes(class, outPath){
    string = "";
    meshes = roiManager("count");
    for (i=0; i<meshes; i++){
    roiManager("select", i);
    Stack.getPosition(channel, slice, frame);
    getSelectionCoordinates(xpoints, ypoints);
    for (j = 0; j < xpoints.length; j++) {
    string = string + "" + xpoints[j] + "," + ypoints[j] + "," + slice + "," + class + "\n";
    }
    class++;
    }
    File.append(string, outPath);
    }
    
    function loadMeshFile(meshPath){
    c = "";
    c = c + toHex(255);
    c = c + toHex(255);
    c = c + toHex(255);
    contents = split(File.openAsString(meshPath), "\n");
    xpoints = newArray();
    ypoints = newArray();
    groups = newArray();
    for (idx=0; idx < contents.length; idx++){
    values = split(contents[idx], ",");
    valuesNext = split(contents[idx+1], ",");
    if (contents[idx+1] == "") {
    valuesNext =  newArray(-1,-1,-1,-1);
    idx++;
    }
    xpoints = Array.concat(xpoints, values[0]);
    ypoints = Array.concat(ypoints, values[1]);
    if (values[3] != valuesNext[3]){
    xpoints = Array.concat(xpoints, xpoints[0]);
    ypoints = Array.concat(ypoints, ypoints[0]);
    groups = Array.concat(groups, values[3]);
    makeSelection("polyline", xpoints, ypoints);
    Roi.setName("Class " + values[3]);
    roiManager("add");
    count = roiManager("count");
    roiManager("select", count-1);
    roiManager("Set Color", c);
    xpoints = newArray();
    ypoints = newArray();
    }
    }
    return groups;
    }
    
    function editMeshes(classVect, numMeshes, outPath){
    waitForUser("Edit the input ROIs if needed\n\nThen click Ok");
    string = "";
    for (i=0; i<numMeshes; i++){
    roiManager("select", i);
    Stack.getPosition(channel, slice, frame);
    getSelectionCoordinates(xpoints, ypoints);
    for (j = 0; j < xpoints.length; j++) {
    string = string + "" + xpoints[j] + "," + ypoints[j] + "," + slice + "," + classVect[i] + "\n";
    groupEnd = classVect[i];
    }
    }
    File.append(string, outPath);
    return groupEnd;
    }
""" % (os.path.join(path, ''), os.path.splitext(imageBaseName)[0])
    macroFid = open(macroPath, 'w')
    macroFid.write(macro)
    macroFid.close()

    args = "-i %s -macro %s" % (image, macroPath)
    viewParams = {showj.ZOOM: 300}
    for key, value in viewParams.items():
        args = "%s --%s %s" % (args, key, value)

    app = "xmipp.ij.commons.XmippImageJ"

    runJavaIJapp(4, app, args).wait()


def lanchIJForViewing(self, path, image):
    self.macroPath = os.path.join(self.path, "View_ROI.ijm")
    imageBaseName = os.path.basename(image)

    roiFile = os.path.join(path, os.path.splitext(imageBaseName)[0] + ".txt")

    macro = r"""path = "%s";
file = "%s"
meshFile = "%s"

outPath = path + meshFile;

// --------- Load SetOfMeshes ---------
if (File.exists(outPath)){
loadMeshFile(outPath);
}

// --------- Functions Definition ---------    
function loadMeshFile(meshPath){
c = "";
c = c + toHex(255*random);
c = c + toHex(255*random);
c = c + toHex(255*random);
contents = split(File.openAsString(meshPath), "\n");
xpoints = newArray();
ypoints = newArray();
groups = newArray();
for (idx=0; idx < contents.length; idx++){
values = split(contents[idx], ",");
if (idx+1 < contents.length){
valuesNext = split(contents[idx+1], ",");
}
else{
valuesNext =  newArray(-1,-1,-1,-1);
}
xpoints = Array.concat(xpoints, values[0]);
ypoints = Array.concat(ypoints, values[1]);
if (values[2] != valuesNext[2]){
xpoints = Array.concat(xpoints, xpoints[0]);
ypoints = Array.concat(ypoints, ypoints[0]);
groups = Array.concat(groups, values[3]);
Stack.setSlice(values[2]);
makeSelection("polyline", xpoints, ypoints);
Roi.setName("Class " + values[3]);
Roi.setStrokeWidth(5);
roiManager("add");
count = roiManager("count");
roiManager("select", count-1);
roiManager("Set Color", c);
if (values[3] != valuesNext[3]){
c = "";
c = c + toHex(255*random);
c = c + toHex(255*random);
c = c + toHex(255*random);
}
xpoints = newArray();
ypoints = newArray();
}
}
return groups;
}
""" % (os.path.join(path, ''), os.path.splitext(imageBaseName)[0], roiFile)
    macroFid = open(self.macroPath, 'w')
    macroFid.write(macro)
    macroFid.close()

    args = "-i %s -macro %s" % (image, self.macroPath)
    viewParams = {showj.ZOOM: 50}
    for key, value in viewParams.items():
        args = "%s --%s %s" % (args, key, value)

    app = "xmipp.ij.commons.XmippImageJ"

    runJavaIJapp(4, app, args).wait()