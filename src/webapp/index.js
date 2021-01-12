var loadModel = _.once(function() {
    //tf.disposeVariables();
    return tf.loadLayersModel("data/model/model.json");
})

const class_labels = ['Golden Retriever','Non-Golden Dog','Simba'];



jQuery(document).ready(function($) {
    var speechElement = $("#prediction-result");
    var reader = new FileReader();

    $("#image-selector").change(function () {
        $("#prediction-result").empty();
        reader.onload = function () {
            let dataURL = reader.result;
            $("#selected-image").attr("src", dataURL);
        }
    
        let file = $("#image-selector").prop('files')[0];
        reader.readAsDataURL(file);
    });
    

    $("#prediction-generate").click(function() {

        speechElement.text("Loading model, generating prediction...");

        loadModel().then(function(model) {
            console.log(model);
            console.log('Status: Model loaded');
            //const img = new Image(height="300", width="300");
            //img.src = reader.result;
            // console.log("Reader: ", reader);
            let img = $("#selected-image").get(0);
            var a1 = tf.browser.fromPixels(img); // Used image must have defined attribute height and width, otherwise tf.browser.fromPixels throws an error.
            const resized = tf.image.resizeBilinear(a1, [300, 300]).toFloat();
            const offset = tf.scalar(255.0);
            // const normalized  = resized.sub(offset).div(offset);
            const normalized = tf.scalar(1.0).sub(resized.div(offset));
            const a3 = normalized.expandDims(0)

            //const normalized  = tf.resized.sub(offset).div(offset);
            //var a2 = a1.expandDims(0); // Create a new variable as reshaping the existing one does not work.          
            //var a3 = tf.image.resizeBilinear(a2,[300,300]);
            console.log('Tensor Shape: ' + a3.shape);
            
            
            var prediction = model.predict(a3);
            console.log(prediction);
            var result = prediction.dataSync();
            var result_tensor = tf.tensor1d(result);
            var result_class_index = result_tensor.argMax(0).arraySync();
            var result_class = class_labels[result_class_index];

            console.log(result);
            console.log(result_class_index);
            console.log(result_class);

            const output = "Classes: " + class_labels  + " <br />" +  "Class Probabilities: " + result + " <br />" + "\Therefore the predicted class is: " + result_class;
            speechElement.html(output);
        })
    })
});