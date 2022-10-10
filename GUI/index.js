$().ready(() => {
    empty_table()

    $("#upload-image").change(load_image)

    $("#masks-opacity").on("change", function() {
        $("#image-masks").css("opacity",$("#masks-opacity").val()/100)
    });
    $("#image-masks").css("opacity",$("#masks-opacity").val()/100)

    $("#show-centers").change(function() {
        if ($("#show-centers").is(':checked') && $("#image-centers").attr("src") != ""){
            $("#image-centers").show()
        }
        else {
            $("#image-centers").hide()
        }
    });

    $("#show-masks").change(function() {
        if ($("#show-masks").is(':checked') && $("#image-masks").attr("src") != ""){
            $("#image-masks").show()
        }
        else {
            $("#image-masks").hide()
        }
    });

    $("#select-x").change(get_plot)
    $("#select-y").change(get_plot)
})

var request_count = 0


function generate_result_table(headers, data) {
    table = $("#results-table")
    table.empty()
    header = $("<tr></tr>")
    header.addClass("header")
    for (let i = 0; i < headers.length; i++) {
        header.append("<th>" + headers[i] + "</th>")
    }

    table.append(header)

    for (let row = 0; row < data.length; row++) {
        let tr = $("<tr></tr>")
        for (let col = 0; col < data[row].length; col++) {
            tr.append("<td>" + data[row][col] + "</td>")
        }
        table.append(tr)
    }
}

function load_image() {
    files = $("#upload-image").prop("files")
    if (files.length == 0) { return }

    file = files[0]

    width = $("#input-image-width").val();
    $("#input-image-width").prop('disabled', true);

    var formData = new FormData();
    formData.append('file', file);
    formData.append("width", width);
    
    $("#image-file-name").text(file.name)

    $.ajax({
        url: 'upload-image',
        type: 'POST',
        data: formData,
        processData: false,  // tell jQuery not to process the data
        contentType: false,  // tell jQuery not to set contentType
        success: function(data) {
            request_count += 1;
            $("#raw-image").attr("src","images/raw"+`${request_count}`);
            $("#raw-image").show()

            $("#upload-image-label").hide() 
            process_image()
        }
    });
}

function process_image(){
    $.ajax({
        url: 'process-image',
        type: 'POST',
        processData: false,  // tell jQuery not to process the data
        contentType: false,  // tell jQuery not to set contentType
        success: function(data) {
            $("#image-masks").attr("src","images/masks"+`${request_count}`);
            $("#image-masks").show()
            $("#image-centers").attr("src","images/centers"+`${request_count}`);
            $("#image-centers").show()

            data = JSON.parse(data)

            $("#data-cell-count").text(data.cells)//.text(data["cells"])
            $("#data-confluency").text(data.confluency)

            update_table(data.regionprops)

            get_plot();
        }
    });
}

function get_plot(){
    xaxis = $("#select-x").val()
    yaxis = $("#select-y").val()

    data = {"xaxis": xaxis, "yaxis":yaxis}

    $.ajax({
        url: 'generate-plot',
        type: 'POST',
        data: JSON.stringify(data),
        contentType: "application/json; charset=utf-8",
        success: function(data) {
            data = JSON.parse(data)

            $("#plot").html(data.plot);
        }
    });
}


const table_header = ["", "Mean", "StdDev", "Min", "Median", "Max"]
const table_props = [["area","Area (&mu;m&#178;)"], ["eccentricity","Eccentricity"], ["major_axis","Long axis (&mu;m)"], ["minor_axis","Short axis (&mu;m)"]]
function update_table(regionprops){

    table = []
    for (let i = 0; i < table_props.length; i++){
        p = regionprops[table_props[i][0]]
        table.push([table_props[i][1], p["mean"], p["stddev"], p["min"], p["median"], p["max"]])
    }

    generate_result_table(table_header,table)
}

function empty_table(){
    table = []
    for (let i = 0; i < table_props.length; i++){
        table.push([table_props[i][1], "","","","",""])
    }

    generate_result_table(table_header,table)
}
