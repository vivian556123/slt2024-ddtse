function createAudioHTML(path, flat) {
  if (flat) {
    return '<audio controls controlslist="nodownload" class="px-1" style="width: 36vw;"> <source src=' +
        path +
        ' type="audio/wav">Your browser does not support the audio element.</audio>';
  }
  return '<audio controls controlslist="nodownload" class="px-1"> <source src=' +
      path +
      ' type="audio/wav">Your browser does not support the audio element.</audio>';
}

function createFigureHTML(path, flat) {
  if (flat) {
    return '<img src=' + path + ' style="width: 9vw;">';
  }
  return '<img src=' + path + '>';
}




function generatemulti_noisyTable(tableId, filenames, page) {
  let numPerPage = 3;
  let table = document.getElementById(tableId + '-operation');
  let nrRows = table.rows.length;
  for (let i = 1; i < nrRows; i++) {
    table.deleteRow(1);
  }
  const prefix = 'multi_noisy_spec_wav/';
  const end_idx = page * numPerPage;
  for (let i = (page - 1) * numPerPage; i < end_idx; i++) {
    // for each i, insert two rows
    //let row = table.insertRow(i % numPerPage + 1);
    let row = table.insertRow((i % numPerPage) *2 + 1);
    let row2 = table.insertRow((i % numPerPage*2) + 2);
    
    row.style.height = '80px';
    //row2.style.height = '80px';
    if (i < filenames.length) {
      let cell = row.insertCell(0);
      let cell2 = row2.insertCell(0)
      let reg = /[0-9]+/g;
      let command = filenames[i].replace(reg,"");
      //cell.innerHTML = command.replaceAll('_', ' ');
      //cell.innerHTML = filenames[i]
      //cell.innerHTML = prefix + filenames[i].replace("samples/","samples/mixture_")+".wav"
      cell.innerHTML = filenames[i]
      cell.style.width = '1vw';
      cell.style.textAlign = "center";
      cell2.innerHTML = " "
      cell2.style.textAlign = "center";

      cell = row.insertCell(1);
      cell2 = row2.insertCell(1)
      cell2.innerHTML = createFigureHTML(prefix+ "mixture_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"mixture_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";

      cell = row.insertCell(2);
      cell2 = row2.insertCell(2)
      cell2.innerHTML = createFigureHTML(prefix+ "gt_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"gt_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";

      cell = row.insertCell(3);
      cell2 = row2.insertCell(3)
      cell2.innerHTML = createFigureHTML(prefix+ "DDCEM_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"DDCEM_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";

      cell = row.insertCell(4);
      cell2 = row2.insertCell(4)
      cell2.innerHTML = createFigureHTML(prefix+ "DPCCN_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"DPCCN_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";

      cell = row.insertCell(5);
      cell2 = row2.insertCell(5)
      cell2.innerHTML = createFigureHTML(prefix+ "RDDCEM_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"RDDCEM_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";
    } else {
      let cell = row.insertCell(0);
      cell.innerHTML = '<br>';
      cell = row.insertCell(1);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
      cell = row.insertCell(2);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
      cell = row.insertCell(3);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
      cell = row.insertCell(4);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
      cell = row.insertCell(5);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
    }
  }
}


function generatemulti_cleanTable(tableId, filenames, page) {
  let numPerPage = 3;
  let table = document.getElementById(tableId + '-operation');
  let nrRows = table.rows.length;
  for (let i = 1; i < nrRows; i++) {
    table.deleteRow(1);
  }
  const prefix = 'multi_clean_spec_wav/';
  const end_idx = page * numPerPage;
  for (let i = (page - 1) * numPerPage; i < end_idx; i++) {
    // for each i, insert two rows
    //let row = table.insertRow(i % numPerPage + 1);
    let row = table.insertRow((i % numPerPage) *2 + 1);
    let row2 = table.insertRow((i % numPerPage*2) + 2);
    
    row.style.height = '80px';
    //row2.style.height = '80px';
    if (i < filenames.length) {
      let cell = row.insertCell(0);
      let cell2 = row2.insertCell(0)
      let reg = /[0-9]+/g;
      let command = filenames[i].replace(reg,"");
      //cell.innerHTML = command.replaceAll('_', ' ');
      //cell.innerHTML = filenames[i]
      //cell.innerHTML = prefix + filenames[i].replace("samples/","samples/mixture_")+".wav"
      cell.innerHTML = filenames[i]
      cell.style.textAlign = "center";
      cell2.innerHTML = " "
      cell2.style.textAlign = "center";

      cell = row.insertCell(1);
      cell2 = row2.insertCell(1)
      cell2.innerHTML = createFigureHTML(prefix+ "mixture_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"mixture_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";

      cell = row.insertCell(2);
      cell2 = row2.insertCell(2)
      cell2.innerHTML = createFigureHTML(prefix+ "gt_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"gt_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";

      cell = row.insertCell(3);
      cell2 = row2.insertCell(3)
      cell2.innerHTML = createFigureHTML(prefix+ "DDCEM_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"DDCEM_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";

      cell = row.insertCell(4);
      cell2 = row2.insertCell(4)
      cell2.innerHTML = createFigureHTML(prefix+ "DPCCN_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"DPCCN_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";

      cell = row.insertCell(5);
      cell2 = row2.insertCell(5)
      cell2.innerHTML = createFigureHTML(prefix+ "RDDCEM_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"RDDCEM_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";
    } else {
      let cell = row.insertCell(0);
      cell.innerHTML = '<br>';
      cell = row.insertCell(1);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
      cell = row.insertCell(2);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
      cell = row.insertCell(3);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
      cell = row.insertCell(4);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
      cell = row.insertCell(5);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
    }
  }
}




function generatesingle_noisyTable(tableId, filenames, page) {
  let numPerPage = 3;
  let table = document.getElementById(tableId + '-operation');
  let nrRows = table.rows.length;
  for (let i = 1; i < nrRows; i++) {
    table.deleteRow(1);
  }
  const prefix = 'single_enh_spec_wav/';
  const end_idx = page * numPerPage;
  for (let i = (page - 1) * numPerPage; i < end_idx; i++) {
    // for each i, insert two rows
    //let row = table.insertRow(i % numPerPage + 1);
    let row = table.insertRow((i % numPerPage) *2 + 1);
    let row2 = table.insertRow((i % numPerPage*2) + 2);
    
    row.style.height = '80px';
    //row2.style.height = '80px';
    if (i < filenames.length) {
      let cell = row.insertCell(0);
      let cell2 = row2.insertCell(0)
      let reg = /[0-9]+/g;
      let command = filenames[i].replace(reg,"");
      //cell.innerHTML = command.replaceAll('_', ' ');
      //cell.innerHTML = filenames[i]
      //cell.innerHTML = prefix + filenames[i].replace("samples/","samples/mixture_")+".wav"
      cell.innerHTML = filenames[i]
      cell.style.textAlign = "center";
      cell2.innerHTML = " "
      cell2.style.textAlign = "center";

      cell = row.insertCell(1);
      cell2 = row2.insertCell(1)
      cell2.innerHTML = createFigureHTML(prefix+ "mixture_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"mixture_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";

      cell = row.insertCell(2);
      cell2 = row2.insertCell(2)
      cell2.innerHTML = createFigureHTML(prefix+ "gt_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"gt_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";

      cell = row.insertCell(3);
      cell2 = row2.insertCell(3)
      cell2.innerHTML = createFigureHTML(prefix+ "DDCEM_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"DDCEM_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";

      cell = row.insertCell(4);
      cell2 = row2.insertCell(4)
      cell2.innerHTML = createFigureHTML(prefix+ "DCCRN_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"DCCRN_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";

      cell = row.insertCell(5);
      cell2 = row2.insertCell(5)
      cell2.innerHTML = createFigureHTML(prefix+ "SGMSE_" + filenames[i]+".png", true)
      cell2.style.textAlign = "center";
      cell.innerHTML = createAudioHTML(prefix +"SGMSE_"+ filenames[i]+".wav", false);
      cell.style.textAlign = "center";
    } else {
      let cell = row.insertCell(0);
      cell.innerHTML = '<br>';
      cell = row.insertCell(1);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
      cell = row.insertCell(2);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
      cell = row.insertCell(3);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
      cell = row.insertCell(4);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
      cell = row.insertCell(5);
      cell.innerHTML = '<br>';
      cell.style.textAlign = "center";
    }
  }
}
noisy_filelist = ["121-121726-0006_61-70970-0003","2300-131720-0041_1284-1180-0023","5142-36377-0005_6930-81414-0019","1089-134686-0007_7021-85628-0018","2300-131720-0036_7176-92135-0006","5105-28240-0017_6930-81414-0015","121-127105-0011_4970-29095-0009","121-127105-0028_4507-16021-0025","6829-68769-0012_1284-1181-0001"]
generatemulti_noisyTable('functionDDTSE', noisy_filelist  , 1);


clean_filelist = ["2830-3980-0072_3575-170457-0021","4992-41806-0015_260-123440-0020","5142-36377-0002_6930-75918-0002","1089-134691-0025_8224-274384-0005","260-123286-0023_2830-3980-0030","8224-274381-0013_1320-122617-0013","121-127105-0011_4970-29095-0009","3575-170457-0009_8555-284447-0023","8555-284447-0004_5683-32879-0000"]
generatemulti_cleanTable('functionDDTSE-clean', clean_filelist  , 1);


enh_filelist =  ['1089-134686-0034_5639-40744-0005','1320-122617-0021_5105-28233-0002','2830-3980-0052_8455-210777-0035']
generatesingle_noisyTable('functionDDTSE-enh', enh_filelist  , 1);


$(document).ready(function() {
  for (let i = 1; i <= 3; i++) {
    let id = '#functionDDTSE-' + i;
    $(id).click(function() {
      generatemulti_noisyTable(
          'functionDDTSE',
          noisy_filelist, i);
      $(id).parent().siblings().removeClass('active');
      $(id).parent().addClass('active');
      return false;
    });
  }
});

$(document).ready(function() {
  for (let i = 1; i <= 3; i++) {
    let id = '#functionDDTSE-clean-' + i;
    $(id).click(function() {
      generatemulti_cleanTable(
          'functionDDTSE-clean',
          clean_filelist, i);
      $(id).parent().siblings().removeClass('active');
      $(id).parent().addClass('active');
      return false;
    });
  }
});

