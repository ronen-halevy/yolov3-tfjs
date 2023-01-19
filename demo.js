import { YoloV3, createModel } from './YoloV3.js';
import Render from './Render.js';

import configModel from './configs/configModel.json' assert { type: 'json' };
import configNms from './configs/configNms.json' assert { type: 'json' };
import configRender from './configs/configRender.json' assert { type: 'json' };

import cocoExamples from './examples/cocoExamples.json' assert { type: 'json' };

const font = configRender.font;
const lineWidth = configRender.lineWidth;
const lineColor = configRender.lineColor;
const textColor = configRender.textColor;
const textBackgoundColor = configRender.textBackgoundColor;

// var model = '';
// var anchors = '';
var yoloV3 = '';
var classNames = '';

const modelsTable = configModel.models;
const models = Object.keys(modelsTable);
var selectedModel = Object.keys(modelsTable)[0];
var selectedWeights = Object.keys(modelsTable[selectedModel])[0];

// const canvas = $('#canvas')[0];
const draw = new Render(
	canvas,
	lineWidth,
	lineColor,
	font,
	textColor,
	textBackgoundColor
);
const loadModel = async (modelsTable, selectedModel, selectedWeights) => {
	$('#waitLoadingModel').show();

	const { modelUrl, anchorsUrl, classNamesUrl } =
		modelsTable[selectedModel][selectedWeights];

	const [model, anchors, classNamesString] = await createModel(
		modelUrl,
		anchorsUrl,
		classNamesUrl
	);
	classNames = classNamesString.split(/\r?\n/);

	$('#waitLoadingModel').hide();

	$('#loadedModelTilte').text(
		'Loaded: ' + selectedModel + '+' + selectedWeights
	);
	return [model, anchors, classNames];
};
const onLoadModel = async () => {
	const [model, anchors, classNames] = await loadModel(
		modelsTable,
		selectedModel,
		selectedWeights
	);
	const nClasses = classNames.length;
	const { scoreTHR, iouTHR, maxBoxes } = configNms;

	yoloV3 = new YoloV3(
		model,
		anchors.anchor,
		nClasses,
		scoreTHR,
		iouTHR,
		maxBoxes
	);
};

$(document).ready(function () {
	$('#waitYolo').hide();
	$('#waitLoadingModel').hide();
	onLoadModel();
	$('#loadModel').text('Load Model');
	$('#loadModel').click(onLoadModel);

	var scaleFactor = 0.125;
	$('#scale').text('x' + scaleFactor);
	$('#scale').click(() => {
		scaleFactor = scaleFactor * 2 > 1 ? 0.125 : scaleFactor * 2;
		$('#scale').text('x' + scaleFactor);
	});

	const onChangeWןwights = (event) => {
		selectedWeights = event.target.value;
	};
	const onChangeModelSelect = (event) => {
		selectedModel = event.target.value;

		createWeightsButtons(selectedModel);
	};
	const createWeightsButtons = (selectedModel) => {
		selectedWeights = Object.keys(modelsTable[selectedModel]);
		$('#divRadioSelectWeights').empty();
		selectedWeights.map((option, index) => {
			$('#divRadioSelectWeights')
				.append(
					$('<input>')
						.prop({
							type: 'radio',
							id: option,
							name: 'weights',
							value: option,
						})
						.change(onChangeWןwights)
				)
				.append(
					$('<label>')
						.prop({
							for: option,
						})
						.text(option)
				)
				.append($('<br>'));
		});
	};

	models.map((option, index) => {
		$('#divRadioSelectModel')
			.append(
				$('<input>')
					.prop({
						type: 'radio',
						id: option,
						name: 'model',
						value: option,
					})
					.change(onChangeModelSelect)
			)
			.append(
				$('<label>')
					.prop({
						for: option,
					})
					.text(option)
			)
			.append($('<br>'));
	});

	$("input[id|='YoloV3Tiny']").attr('checked', true);

	// $('#YoloV3').attr('checked', true);
	createWeightsButtons('YoloV3Tiny');
	$("input[id|='coco']").attr('checked', true);

	const cocoImages = cocoExamples.cocoImages;
	var selectedExample = cocoImages[0];
	var exampleUrl = selectedExample.url;
	$('#selectedExampleTitle').text('Title: ' + selectedExample.title);

	cocoImages.map((option, index) => {
		$('#selectExample').append(new Option(option.url, index));
	});
	$('#selectExample').change((event) => {
		selectedExample = cocoImages[event.target.value];
		exampleUrl = selectedExample.url;

		$('#selectedExampleTitle').text(selectedExample.title);
		// selecteTitle = cocoImages[event.target.value].title;
	});

	$('#runYolo').text('Run Yolo');
	$('#runYolo').click(async () => {
		$('#waitYolo').show();

		var imageObject = new window.Image();
		const res = await fetch(exampleUrl);
		const imageBlob = await res.blob();
		const imageObjectURL = URL.createObjectURL(imageBlob);
		imageObject.src = imageObjectURL;
		imageObject.addEventListener('load', async () => {
			const [selBboxes, scores, classIndices] = await yoloV3.detectFrame(
				imageObject
			);
			console.log(selBboxes);
			console.log('selBboxes');

			draw.renderOnImage(
				imageObject,
				selBboxes,
				scores,
				classIndices,
				classNames,
				imageObject.width * scaleFactor,
				imageObject.height * scaleFactor
			);
		});
		$('#waitYolo').hide();
	});
});
